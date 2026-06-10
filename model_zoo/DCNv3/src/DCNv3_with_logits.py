# DCNv3 with BCEWithLogitsLoss for better numerical stability
# This version removes sigmoid activation and uses logits directly

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
import logging


class DCNv3Logits(BaseModel):
    """DCNv3 with BCEWithLogitsLoss for better numerical stability"""
    
    def __init__(self,
                 feature_map,
                 model_id="DCNv3Logits",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_deep_cross_layers=4,
                 num_shallow_cross_layers=4,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.3,
                 layer_norm=True,
                 batch_norm=False,
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 use_domain_aware_structure=False,
                 **kwargs):
        super(DCNv3Logits, self).__init__(feature_map, model_id=model_id, gpu=gpu, 
                                         embedding_regularizer=embedding_regularizer, 
                                         net_regularizer=net_regularizer, **kwargs)
        
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        
        self.ECN = ExponentialCrossNetwork(input_dim, num_deep_cross_layers, layer_norm, 
                                          batch_norm, deep_net_dropout, num_heads)
        self.LCN = LinearCrossNetwork(input_dim, num_shallow_cross_layers, layer_norm, 
                                     batch_norm, shallow_net_dropout, num_heads)
        
        self.use_domain_aware_structure = use_domain_aware_structure
        
        if self.use_domain_aware_structure:
            self._init_domain_aware_structure_params_pytorch(input_dim)
        
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def get_output_activation(self, task):
        """Override to use Identity activation for logits"""
        return nn.Identity()  # Return logits directly
    
    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        if self.use_domain_aware_structure:
            xld_intermediate = self.ECN(feature_emb)
            xls_intermediate = self.LCN(feature_emb)

            xld_flat = xld_intermediate.view(xld_intermediate.size(0), -1)
            xls_flat = xls_intermediate.view(xls_intermediate.size(0), -1)

            self.logits_xld = self._generate_domain_aware_logits_pytorch(X, xld_flat)
            self.logits_xls = self._generate_domain_aware_logits_pytorch(X, xls_flat)
        else:
            self.logits_xld = self.ECN(feature_emb).mean(dim=1)
            self.logits_xls = self.LCN(feature_emb).mean(dim=1)
        
        # Return logits directly (no sigmoid activation)
        logit = (self.logits_xld + self.logits_xls) * 0.5
        
        return_dict = {"y_pred": logit,
                       "y_d": self.logits_xld,
                       "y_s": self.logits_xls}
        return return_dict

    def add_loss(self, return_dict, y_true):
        """Use BCEWithLogitsLoss for better numerical stability"""
        y_pred = return_dict["y_pred"]
        y_d = return_dict["y_d"]
        y_s = return_dict["y_s"]
        
        # Use BCEWithLogitsLoss which is more numerically stable
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')
        loss_d = F.binary_cross_entropy_with_logits(y_d, y_true, reduction='mean')
        loss_s = F.binary_cross_entropy_with_logits(y_s, y_true, reduction='mean')
        
        # Check for NaN/Inf values
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: NaN or Inf detected in main loss: {loss}")
            loss = torch.zeros_like(loss)
            
        if torch.isnan(loss_d).any() or torch.isinf(loss_d).any():
            print(f"Warning: NaN or Inf detected in deep loss: {loss_d}")
            loss_d = torch.zeros_like(loss_d)
            
        if torch.isnan(loss_s).any() or torch.isinf(loss_s).any():
            print(f"Warning: NaN or Inf detected in shallow loss: {loss_s}")
            loss_s = torch.zeros_like(loss_s)
        
        weight_d = loss_d - loss
        weight_s = loss_s - loss
        
        # Safe handling of weights
        weight_d = torch.where(torch.isnan(weight_d), torch.zeros_like(weight_d), weight_d)
        weight_s = torch.where(torch.isnan(weight_s), torch.zeros_like(weight_s), weight_s)
        
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros_like(weight_d))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros_like(weight_s))
        
        total_loss = loss + loss_d * weight_d + loss_s * weight_s
        
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            print(f"Warning: NaN or Inf detected in total loss, using fallback")
            total_loss = loss
            
        return total_loss


# Import the necessary components from the original DCNv3
from model_zoo.DCNv3.src.DCNv3 import ExponentialCrossNetwork, LinearCrossNetwork 