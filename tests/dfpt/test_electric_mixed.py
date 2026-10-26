"""Full-root electric enthalpy, Maxwell symmetry and directional chi response."""
import jax
import jax.numpy as jnp
import numpy as np

from test_electric_field import dimer, energy_and_fock


def test_dimer_full_root_maxwell_and_mixed_susceptibility():
    from gradscf.dfpt.electric import _electric_model, _solve_electric
    from gradscf.solvers import LinearSolverConfig
    from gradscf.solvers.nonlinear.iterate import NonlinearConfig
    data, state, counts, _, _, chi_reference = dimer((4, 1, 1))
    model, diagonal = _electric_model(data, state, nocc=counts, evaluate=energy_and_fock)
    linear = LinearSolverConfig(method='direct', rtol=1e-11, atol=1e-12, max_dense=128)
    config = NonlinearConfig(maxiter=80, residual_tol=1e-11, energy_tol=1e-13, damping=.2)

    def solve(coordinate, field):
        hopping = jnp.array([[0., 1.], [1., 0.]])
        varied = data._replace(hcore=data.hcore+coordinate*hopping)
        return _solve_electric(model, diagonal, (varied, None, field),
                               config=config, linear_config=linear)

    energy = lambda x,e:solve(x,e)[1]['electric_enthalpy']
    polar = lambda x,e:solve(x,e)[1]['polarization']
    x, field = jnp.array(0.), jnp.zeros(3)
    volume = float(np.linalg.det(np.asarray(data.lattice)))
    p0 = polar(x,field)
    gradient = jax.jit(jax.grad(energy, argnums=1))(x,field)
    curvature = jax.jit(jax.jacfwd(jax.grad(energy,argnums=1),argnums=1))(x,field)
    maxwell = jax.jit(jax.jacfwd(lambda e:-jax.grad(energy,argnums=0)(x,e)))(field)
    chi = jax.jacfwd(polar,argnums=1)
    mixed = jax.jit(lambda r:jax.jvp(lambda t:chi(t,field),(r,),(jnp.ones_like(r),))[1])(x)

    # Independent closed-form valence occupation for [[a,t],[t,-a]]. Its
    # derivatives contain no SCF/projector/implicit-root calculation.
    a, t = .4, .3
    radius = np.sqrt(a*a+t*t)
    occupation = (1+a/radius)/2
    occupation_derivative = -a*t/(2*radius**3)
    lattice = np.asarray(data.lattice)
    exponent = np.exp(-2j*np.pi*np.array([.15,.55])/4)
    link = (1-occupation)*exponent[0]+occupation*exponent[1]
    quotient = (exponent[1]-exponent[0])/link
    b = 4*np.imag(quotient)*lattice[0]/(2*np.pi*volume)
    db = 4*np.imag(-quotient**2*occupation_derivative)*lattice[0]/(2*np.pi*volume)
    prefactor = volume*t*t/(4*radius**3)
    dprefactor = volume/4*(2*t/radius**3-3*t**3/radius**5)
    expected_mixed = dprefactor*np.outer(b,b)+prefactor*(np.outer(db,b)+np.outer(b,db))
    expected_maxwell = volume*b*occupation_derivative
    np.testing.assert_allclose(gradient,-volume*p0,atol=2e-12)
    np.testing.assert_allclose(curvature,-volume*chi_reference,atol=2e-11,rtol=2e-9)
    np.testing.assert_allclose(maxwell,expected_maxwell,atol=2e-11,rtol=2e-9)
    np.testing.assert_allclose(mixed,expected_mixed,atol=2e-11,rtol=2e-9)
    respond = jax.jit(chi)
    for step in (3e-4,1e-4):
        plus, minus = solve(step,field)[0], solve(-step,field)[0]
        assert plus.converged & minus.converged
        assert max(float(plus.residual_norm),float(minus.residual_norm)) < 1e-11
        finite = (respond(jnp.array(step),field)-respond(jnp.array(-step),field))/(2*step)
        np.testing.assert_allclose(mixed,finite,atol=2e-8,rtol=2e-5)
    print('Dimer Maxwell',np.asarray(maxwell).tolist(),'mixed chi',np.asarray(mixed).tolist(),
          'analytic mixed error',float(np.max(abs(mixed-expected_mixed))),flush=True)


def test_periodic_h2_full_root_maxwell_and_directional_dielectric_mode():
    from gradscf.pbc import gto, scf
    from gradscf.pbc._kpoint import energy_and_fock as evaluate_hf
    from gradscf.pbc.scf import run_kpoint_scf
    from gradscf.integrals.periodic.fft import build_inputs
    from gradscf.dfpt.electric import _electric_model, _field_problem, _solve_electric, born_problem
    from gradscf.dfpt.atomic import atomic_problem
    from gradscf.solvers import LinearSolverConfig
    from gradscf.solvers.nonlinear.iterate import NonlinearConfig
    cell = gto.M(atom='H .4 .5 .6; H 1.8 .8 .9',a=np.eye(3)*6.,unit='Bohr',mesh=(9,)*3)
    controls = dict(conv_tol=1e-12,conv_tol_density=1e-11,conv_tol_grad=1e-11,max_cycle=150,damping=.1)
    mf = scf.KRHF(cell,kpts=cell.make_kpts((3,1,1))).run(**controls)
    assert mf.converged
    evaluate = lambda d,a,p:evaluate_hf(d,a,mesh=cell.mesh,xc='hf',exxdiv='ewald')
    model, diagonal = _electric_model(mf.inputs,mf.result,nocc=cell.nelec,evaluate=evaluate)
    linear = LinearSolverConfig(method='direct',rtol=1e-11,atol=1e-12,max_dense=128)
    config = NonlinearConfig(maxiter=150,residual_tol=1e-11,energy_tol=1e-13,damping=.2)
    displacement = jnp.zeros_like(cell.coords).at[1,0].set(1.)
    direction = jnp.array([1.,0.,0.])

    def inputs_at(t):
        return build_inputs(cell,parameters=cell.parameters_at(cell.coords+t*displacement),kpts=mf.kpts)

    def solve(t,e):
        return _solve_electric(model,diagonal,(inputs_at(t),None,e*direction),
                               config=config,linear_config=linear)

    energy = lambda t,e:solve(t,e)[1]['electric_enthalpy']
    polarization = lambda t,e:solve(t,e)[1]['polarization'][0]
    zero = jnp.array(0.)
    volume = float(cell.volume)
    derivatives = jax.jit(jax.grad(energy,argnums=(0,1)))(zero,zero)
    field_second = jax.jit(jax.grad(jax.grad(energy,argnums=1),argnums=1))(zero,zero)
    force = jax.jit(lambda t,e:-jax.grad(energy,argnums=0)(t,e))
    maxwell = jax.jit(jax.grad(force,argnums=1))(zero,zero)
    susceptibility = jax.grad(polarization,argnums=1)
    mixed = jax.jit(jax.grad(susceptibility,argnums=0))(zero,zero)
    atoms = atomic_problem(mf)
    reference_force = atoms.quantities(atoms.solution,cell.coords)['forces'][1,0]
    born = born_problem(mf).response(tangent=displacement,config=linear)
    electric = _field_problem(model).response(tangent=direction,config=linear)
    assert born.converged & electric.converged
    np.testing.assert_allclose(-derivatives[0],reference_force,atol=2e-10,rtol=2e-8)
    np.testing.assert_allclose(derivatives[1],-volume*polarization(zero,zero),atol=2e-10)
    np.testing.assert_allclose(field_second,-volume*electric.values['polarization'][0],atol=2e-9)
    np.testing.assert_allclose(maxwell,volume*born.values['polarization'][0],atol=2e-9)

    # Rebuild and cold-converge the ordinary zero-field SCF at each displacement.
    # Its new canonical reference/phase chart independently gives chi for FD.
    cold_scf = jax.jit(lambda t:run_kpoint_scf(inputs_at(t),mesh=cell.mesh,
        nelec=cell.nelec,xc='hf',exxdiv='ewald',**controls))
    errors = []
    for step in (3e-4,1e-4):
        values = []
        for sign in (1.,-1.):
            t = jnp.array(sign*step)
            data = inputs_at(t)
            state = cold_scf(t)
            assert state.converged
            rebuilt,_ = _electric_model(data,state,nocc=cell.nelec,evaluate=evaluate)
            response = _field_problem(rebuilt).response(tangent=direction,config=linear)
            assert response.converged
            values.append(response.values['polarization'][0])
        finite = (values[0]-values[1])/(2*step)
        errors.append(float(abs(finite-mixed)))
        np.testing.assert_allclose(mixed,finite,atol=2e-7,rtol=2e-5)
        force_fd = (force(zero,step)-force(zero,-step))/(2*step)
        np.testing.assert_allclose(maxwell,force_fd,atol=2e-7,rtol=2e-5)
    print('H2 Maxwell',float(maxwell),'dchi_xx/dR_Hx',float(mixed),
          'cold-SCF two-step errors',errors,flush=True)


def test_fixed_berry_anchor_gauge_preserves_mixed_response():
    """Change only the orbital frame, retaining density and the same branch."""
    from gradscf.dfpt.electric import _electric_model, _solve_electric
    from gradscf.pbc.polarization import make_plan, polarization
    from gradscf.solvers import LinearSolverConfig
    from gradscf.solvers.nonlinear.iterate import NonlinearConfig
    data,state,counts,_,_,expected = dimer((2,1,1),core=True,transformed=True)
    plan = make_plan(np.asarray(data.kpoints)@np.asarray(data.lattice).T/(2*np.pi))
    reference = polarization(data,state,plan,nocc=counts)
    anchor = tuple(p+2*np.pi*(i+1) for i,p in enumerate(reference.phases))
    volume = float(np.linalg.det(np.asarray(data.lattice)))
    shift = np.arange(1,4)@np.asarray(data.lattice)/volume
    coefficients = np.asarray(state.mo_coeff_spin).copy()
    rng = np.random.default_rng(20261016)
    for spin,no in enumerate(counts):
        for start,stop in [(0,no),(no,coefficients.shape[-1])]:
            n = stop-start
            unitary = np.linalg.qr(rng.normal(size=(len(data.kpoints),n,n))+
                1j*rng.normal(size=(len(data.kpoints),n,n)))[0]
            coefficients[spin,:,:,start:stop] = coefficients[spin,:,:,start:stop]@unitary
    rotated = state._replace(mo_coeff_spin=jnp.asarray(coefficients))
    linear = LinearSolverConfig(method='direct',rtol=1e-11,atol=1e-12,max_dense=128)
    config = NonlinearConfig(maxiter=80,residual_tol=1e-11,energy_tol=1e-13,damping=.2)
    direction = jnp.array([0.,1.,0.])
    results = []
    for supplied in (state,rotated):
        model,diagonal = _electric_model(data,supplied,nocc=counts,evaluate=energy_and_fock,
                                         reference_phases=anchor)
        def observe(t,field):
            varied = data._replace(hcore=data.hcore*(1+t))
            return _solve_electric(model,diagonal,(varied,None,field),
                config=config,linear_config=linear)[1]['polarization']
        def response():
            zero = jnp.array(0.)
            field = jnp.zeros(3)
            chi = lambda t:jax.jvp(lambda e:observe(t,e),(field,),(direction,))[1]
            first,mixed = jax.jvp(chi,(zero,),(jnp.array(1.),))
            return observe(zero,field),first,mixed
        values = jax.jit(response)()
        results.append(values)
        np.testing.assert_allclose(values[0],reference.total+shift,atol=2e-11)
        np.testing.assert_allclose(values[1],expected@direction,atol=2e-11,rtol=2e-9)
        np.testing.assert_allclose(values[2],-expected@direction,atol=2e-11,rtol=2e-9)
    for first,second in zip(*results):
        np.testing.assert_allclose(first,second,atol=2e-11,rtol=2e-9)
    import pytest
    with pytest.raises(ValueError,match='Reference phases'):
        _electric_model(data,state,nocc=counts,evaluate=energy_and_fock,
                        reference_phases=(jnp.zeros((1,20)),)*3)
