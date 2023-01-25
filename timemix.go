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

func (m *TimeMix) Forward(x ag.Node, state *LayerState) ag.Node {
	r, k, v := m.mixWithPreviousTimeStep(x, state)
	y := m.calculateOutput(r, k, v, state)
	m.updateState(x, k, v, state)
	return y
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
