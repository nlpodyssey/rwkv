// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkv

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &TimeMix{}

// TimeMix is a model that implements the TimeMix component.
type TimeMix struct {
	nn.Module

	Key        nn.Param `spago:"type:weights"`
	Value      nn.Param `spago:"type:weights"`
	Receptance nn.Param `spago:"type:weights"`
	Output     nn.Param `spago:"type:weights"`

	TimeDecay nn.Param `spago:"type:weights"`
	TimeFirst nn.Param `spago:"type:weights"`
	TimeMixK  nn.Param `spago:"type:weights"`
	TimeMixV  nn.Param `spago:"type:weights"`
	TimeMixR  nn.Param `spago:"type:weights"`

	Config Config
}

func init() {
	gob.Register(&TimeMix{})
}

func NewTimeMix[T float.DType](c Config, _ int) *TimeMix {
	return &TimeMix{
		Config:     c,
		Key:        nn.NewParam(mat.NewEmptyDense[T](c.DModel, c.DModel)),
		Value:      nn.NewParam(mat.NewEmptyDense[T](c.DModel, c.DModel)),
		Receptance: nn.NewParam(mat.NewEmptyDense[T](c.DModel, c.DModel)),
		Output:     nn.NewParam(mat.NewEmptyDense[T](c.DModel, c.DModel)),
		TimeDecay:  nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
		TimeFirst:  nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
		TimeMixK:   nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
		TimeMixV:   nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
		TimeMixR:   nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
	}
}

// ForwardSingle performs the forward step for a single input.
func (m *TimeMix) ForwardSingle(x ag.Node, state *LayerState) ag.Node {
	r, k, v := m.mixWithPreviousTimeStep(x, state) // state unchanged
	y := m.calculateOutput(r, k, v, state)         // state unchanged
	m.updateState(x, k, v, state)
	return y
}

// mixWithPreviousTimeStep mixes the current input with the previous one.
func (m *TimeMix) mixWithPreviousTimeStep(x ag.Node, state *LayerState) (r, k, v ag.Node) {
	xx := state.AttXX
	xk := ag.Add(ag.Prod(m.TimeMixK, x), ag.Prod(ag.ReverseSub(m.TimeMixK, one), xx))
	xv := ag.Add(ag.Prod(m.TimeMixV, x), ag.Prod(ag.ReverseSub(m.TimeMixV, one), xx))
	xr := ag.Add(ag.Prod(m.TimeMixR, x), ag.Prod(ag.ReverseSub(m.TimeMixR, one), xx))

	k = ag.Mul(m.Key, xk)
	v = ag.Mul(m.Value, xv)
	r = ag.Sigmoid(ag.Mul(m.Receptance, xr))
	return
}

// calculateOutput calculates the output of the time-mix.
func (m *TimeMix) calculateOutput(r, k, v ag.Node, state *LayerState) ag.Node {
	aa, bb, pp := state.AttAA, state.AttBB, state.AttPP

	ww := ag.Add(k, m.TimeFirst)
	p := ag.Max(pp, ww)
	e1 := ag.Exp(ag.Sub(pp, p))
	e2 := ag.Exp(ag.Sub(ww, p))
	a := ag.Add(ag.Prod(e1, aa), ag.Prod(e2, v))
	b := ag.Add(ag.Prod(e1, bb), e2)
	rwkv := ag.Prod(r, ag.Div(a, b))

	return ag.Mul(m.Output, rwkv)
}

// updateState updates the state of the layer with the current time step.
func (m *TimeMix) updateState(x, k, v ag.Node, state *LayerState) {
	aa, bb, pp := state.AttAA, state.AttBB, state.AttPP

	ww := ag.Add(pp, m.TimeDecay)
	p := ag.Max(ww, k)
	e1 := ag.Exp(ag.Sub(ww, p))
	e2 := ag.Exp(ag.Sub(k, p))

	state.AttXX = x
	state.AttAA = ag.Add(ag.Prod(e1, aa), ag.Prod(e2, v))
	state.AttBB = ag.Add(ag.Prod(e1, bb), e2)
	state.AttPP = p
}

// ForwardSequence performs the forward step for a sequence of inputs.
// The state is updated at the end of the sequence.
func (m *TimeMix) ForwardSequence(x []ag.Node, state *LayerState) []ag.Node {
	r, k, v := m.mixWithPreviousTimeSteps(x, state) // state is updated here
	return m.calculateOutputs(r, k, v, state)       // state is updated here
}

// mixWithPreviousTimeStep mixes the current input with the previous one.
func (m *TimeMix) mixWithPreviousTimeSteps(x []ag.Node, state *LayerState) (r, k, v []ag.Node) {
	xx := append([]ag.Node{state.AttXX}, x[:len(x)-1]...)

	k = make([]ag.Node, len(x))
	v = make([]ag.Node, len(x))
	r = make([]ag.Node, len(x))

	// precompute coefficients
	tmk := ag.ReverseSub(m.TimeMixK, one)
	tmv := ag.ReverseSub(m.TimeMixV, one)
	tmr := ag.ReverseSub(m.TimeMixR, one)

	for i, xi := range x {
		xk := ag.Add(ag.Prod(m.TimeMixK, xi), ag.Prod(tmk, xx[i]))
		xv := ag.Add(ag.Prod(m.TimeMixV, xi), ag.Prod(tmv, xx[i]))
		xr := ag.Add(ag.Prod(m.TimeMixR, xi), ag.Prod(tmr, xx[i]))
		k[i] = ag.Mul(m.Key, xk)
		v[i] = ag.Mul(m.Value, xv)
		r[i] = ag.Sigmoid(ag.Mul(m.Receptance, xr))
	}

	state.AttXX = x[len(x)-1]
	return
}

// calculateOutput calculates the output of the time-mix.
func (m *TimeMix) calculateOutputs(r, k, v []ag.Node, state *LayerState) []ag.Node {
	aa, bb, pp := state.AttAA, state.AttBB, state.AttPP

	wkv := make([]ag.Node, len(r))
	for i := 0; i < len(r); i++ {
		ww := ag.Add(k[i], m.TimeFirst)
		p := ag.Max(pp, ww)
		e1 := ag.Exp(ag.Sub(pp, p))
		e2 := ag.Exp(ag.Sub(ww, p))
		a := ag.Add(ag.Prod(e1, aa), ag.Prod(e2, v[i]))
		b := ag.Add(ag.Prod(e1, bb), e2)
		wkv[i] = ag.Div(a, b)

		// update intermediate values
		ww = ag.Add(pp, m.TimeDecay)
		p = ag.Max(ww, k[i])
		e1 = ag.Exp(ag.Sub(ww, p))
		e2 = ag.Exp(ag.Sub(k[i], p))
		aa = ag.Add(ag.Prod(e1, aa), ag.Prod(e2, v[i]))
		bb = ag.Add(ag.Prod(e1, bb), e2)
		pp = p
	}

	out := make([]ag.Node, len(r))
	for i, wkvi := range wkv {
		out[i] = ag.Mul(m.Output, ag.Prod(r[i], wkvi))
	}

	// update state with last computed values
	state.AttAA = aa
	state.AttBB = bb
	state.AttPP = pp

	return out
}
