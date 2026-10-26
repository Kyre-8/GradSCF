"""Thin MACE/Flax NNX adapter with element-specific radial contraction heads."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gradscf.data.molecule import atomic_number
from .basis import supported_elements


class MACEBasisModel(nnx.Module):
    """MACE invariants produce per-atom basis parameters.

    qvszps predicts full raw s/p/d coefficients, with a trainable reference
    initialization bias and no fixed baseline in forward(). The older szp3
    mode predicts bounded tangent coordinates. The upstream energy head is
    unused. Zero output kernels recover the reference basis in either mode.
    """
    def __init__(self,*,elements=None,channels=32,num_interactions=2,max_ell=2,
                 correlation=2,cutoff=5.,zero_init=True,basis_family='szp442_direct',rngs):
        try:
            from e3nn_jax import Irreps
            from mace_jax.modules.models import MACE
            from mace_jax.modules.blocks import RealAgnosticInteractionBlock,RealAgnosticResidualInteractionBlock
        except ImportError as exc:
            raise ImportError('MACEBasisModel requires the pinned mace_jax stack; install GradSCF[nnao] or the vendored src/gradscf/model/nnao/vendor/mace-jax project in a compatible environment.') from exc
        if basis_family not in {'szp3','szp3_direct','szp442_direct','szp663_direct','qvszps'}:raise ValueError('Unknown basis_family.')
        self.basis_family=basis_family
        self.output_shape={'szp3':(2,2),'szp3_direct':(2,3),'szp442_direct':(3,4),'szp663_direct':(3,6),'qvszps':(3,5)}[basis_family]
        supported=tuple(atomic_number(s) for s in supported_elements())
        self.elements=supported if elements is None else tuple(elements)
        if not self.elements or len(set(self.elements))!=len(self.elements) or any(z not in supported for z in self.elements):
            raise ValueError('Model elements must be unique supported main-group atomic numbers.')
        if channels<1 or num_interactions<1 or max_ell<0 or cutoff<=0:
            raise ValueError('Invalid MACE dimensions or cutoff.')
        self.channels=int(channels);self.cutoff=float(cutoff)
        irreps=Irreps(' + '.join(f'{channels}x{l}{"e" if l%2==0 else "o"}' for l in range(max_ell+1)))
        self.feature_width=irreps.dim*(num_interactions-1)+channels
        self.backbone=MACE(r_max=cutoff,num_bessel=8,num_polynomial_cutoff=5,max_ell=max_ell,
            interaction_cls=RealAgnosticResidualInteractionBlock,interaction_cls_first=RealAgnosticInteractionBlock,
            atomic_energies=np.zeros((1,len(self.elements))),atomic_numbers=self.elements,
            num_elements=len(self.elements),num_interactions=num_interactions,hidden_irreps=irreps,
            MLP_irreps=Irreps(f'{channels}x0e'),avg_num_neighbors=4.,correlation=correlation,
            radial_MLP=(32,32),rngs=rngs)
        self.hidden=nnx.Linear(channels+2,channels,param_dtype=jnp.float64,rngs=rngs)
        width=int(np.prod(self.output_shape))
        shape=(len(self.elements),channels,width)
        kernel=jnp.zeros(shape) if zero_init else .01*jax.random.normal(rngs.params(),shape,dtype=jnp.float64)
        self.head_kernel=nnx.Param(kernel)
        bias=np.zeros((len(self.elements),*self.output_shape))
        mask=np.zeros_like(bias)
        if basis_family=='qvszps':
            from .grimme import grimme_templates
            by_z={atomic_number(s):b for s,(b,ecp) in grimme_templates().items()}
            for i,z in enumerate(self.elements):
                for block in by_z[z]:
                    l=block[0];c=np.asarray([r[1] for r in block[1:]])
                    bias[i,l,:len(c)]=c/np.linalg.norm(c)
                    mask[i,l,:len(c)]=1.
        elif basis_family in {'szp3_direct','szp442_direct','szp663_direct'}:
            from .initialization import _direct_templates,_direct_slot
            by_z={atomic_number(s):data['shells'] for s,data in _direct_templates(basis_family).items()}
            for i,z in enumerate(self.elements):
                for shell in by_z[z]:
                    slot=_direct_slot(shell)
                    if slot>=0:
                        c=np.asarray(shell['coefficients'])
                        bias[i,slot,:len(c)]=c/np.linalg.norm(c)
                        mask[i,slot,:len(c)]=1.
        else:
            pblock=set(atomic_number(s) for s in 'B C N O F Ne Al Si P S Cl Ar Ga Ge As Se Br Kr In Sn Sb Te I Xe'.split())
            for i,z in enumerate(self.elements):
                mask[i,0,:]=1.;mask[i,1,:]=float(z in pblock)
        # The bias is trainable. In qvszps mode this is the entire initial
        # coefficient prediction, not a fixed baseline added in forward().
        self.head_bias=nnx.Param(jnp.asarray(bias.reshape(len(self.elements),width)))
        self.output_mask=jnp.asarray(mask)

    def __call__(self,graph):
        if graph.element_order!=self.elements:
            raise ValueError('Graph element_order differs from the MACE model.')
        if graph.cutoff!=self.cutoff:
            raise ValueError('Graph cutoff differs from the MACE model.')
        features=self.backbone(dict(graph),compute_force=False,compute_stress=False,compute_node_feats=True)['node_feats']
        if features.shape[-1]!=self.feature_width:
            raise ValueError('Unexpected upstream MACE feature layout; check the pinned version.')
        # Our configured MACE ends in exactly channels x 0e. Earlier layers
        # contain non-scalars and must never be fed directly to a scalar MLP.
        scalars=features[:,-self.channels:]
        conditions=jnp.stack((graph['total_charge'][graph['batch']],graph['spin'][graph['batch']]),axis=-1)
        h=jax.nn.silu(self.hidden(jnp.concatenate((scalars,conditions),axis=-1)))
        species=graph['node_type']
        output=jnp.einsum('ni,nij->nj',h,self.head_kernel[species])+self.head_bias[species]
        return output.reshape((-1,*self.output_shape))*self.output_mask[species]

    def assemble(self,layout,graph):
        """Bind coefficients and graph coordinates to a matching single-molecule layout."""
        from gradscf.data.molecule import ANGSTROM_TO_BOHR
        if graph['ptr'].shape!=(2,) or len(layout.symbols)!=graph['positions'].shape[0]:
            raise ValueError('assemble requires one graph matching the basis layout.')
        expected=np.asarray([self.elements.index(atomic_number(s)) for s in layout.symbols])
        actual=graph['node_type']
        if not isinstance(actual,jax.core.Tracer) and not np.array_equal(np.asarray(actual),expected):
            raise ValueError('Atom ordering differs between graph and basis layout.')
        parameters=layout.bind(self(graph),coords_bohr=graph['positions']*ANGSTROM_TO_BOHR)
        # Preserve a fail-closed numerical signal for a mismatched traced graph.
        valid=jnp.all(actual==jnp.asarray(expected))
        from dataclasses import replace
        return replace(parameters,coefficients=tuple(jnp.where(valid,c,jnp.nan) for c in parameters.coefficients))


__all__=['MACEBasisModel']
