from .standard.transformer import CrossAttentionCfg
from .dual_pathway_attention import SpatialTransformer3DCfg

MultiViewAttentionCfg = CrossAttentionCfg | SpatialTransformer3DCfg
