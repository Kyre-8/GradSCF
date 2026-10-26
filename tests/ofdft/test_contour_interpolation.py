"""Fourier slices used in the scientific example preserve the sampled field."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('matplotlib')


def test_fourier_plane_matches_analytic_density_on_an_oblique_cut():
    path=Path(__file__).resolve().parents[2]/'examples/ofdft/plot_atlas_contours.py'
    spec=importlib.util.spec_from_file_location('plot_atlas_contours',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mesh=(5,7,9)
    grid=np.meshgrid(*[np.arange(n)/n for n in mesh],indexing='ij')
    rho=1+.2*np.cos(2*np.pi*(grid[0]+2*grid[1]-grid[2]))
    lattice=np.array([[2.,0.,0.],[.3,3.,0.],[.2,.1,4.]])
    a=lattice[0]+lattice[1];b=lattice[1]+lattice[2]
    origin=np.array([.1,.2,-.3])@lattice
    plane=module.fourier_plane(rho,lattice,a,b,origin=origin,npoints=64)
    s,t=np.meshgrid(np.arange(64)/64,np.arange(64)/64,indexing='ij')
    expected=1+.2*np.cos(2*np.pi*(3*s+t+.8))
    np.testing.assert_allclose(plane,expected,atol=1e-13)
