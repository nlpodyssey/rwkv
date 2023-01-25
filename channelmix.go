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

// ChannelMix implements the channel mix module.
type ChannelMix struct {
	nn.Module

	Key        nn.Param `spago:"type:weights"`
	Value      nn.Param `spago:"type:weights"`
	Receptance nn.Param `spago:"type:weights"`

	TimeMixK nn.Param `spago:"type:weights"`
	TimeMixR nn.Param `spago:"type:weights"`
}

func init() {
	gob.Register(&ChannelMix{})
}

func NewChannelMix[T float.DType](c Config, _ int) *ChannelMix {
	hidden := 4 * c.DModel
	return &ChannelMix{
		Key:        nn.NewParam(mat.NewEmptyDense[T](hidden, c.DModel)),
		Value:      nn.NewParam(mat.NewEmptyDense[T](c.DModel, hidden)),
		Receptance: nn.NewParam(mat.NewEmptyDense[T](c.DModel, c.DModel)),
		TimeMixK:   nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
		TimeMixR:   nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
	}
}

func (m *ChannelMix) Forward(x ag.Node, state *LayerState) (rkv ag.Node) {
	xx := state.FfnXX
	xk := ag.Add(ag.Prod(x, m.TimeMixK), ag.Prod(ag.ReverseSub(m.TimeMixK, one), xx))
	xr := ag.Add(ag.Prod(x, m.TimeMixR), ag.Prod(ag.ReverseSub(m.TimeMixR, one), xx))
	state.FfnXX = x

	k := ag.Mul(m.Key, xk)
	k = ag.Square(ag.ReLU(k))
	kv := ag.Mul(m.Value, k)
	rkv = ag.Prod(ag.Sigmoid(ag.Mul(m.Receptance, xr)), kv)
	return
}
