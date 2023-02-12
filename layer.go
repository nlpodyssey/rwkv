// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkv

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

var one = ag.Scalar(1.0)

// Layer is a single block of the RWKV model.
type Layer struct {
	nn.Module

	LN0 *layernorm.Model
	LN1 *layernorm.Model
	LN2 *layernorm.Model

	ChanMix *ChannelMix
	TimeMix *TimeMix

	ID int
}

func init() {
	gob.Register(&Layer{})
}

// NewLayer returns a new RWKV layer.
func NewLayer[T float.DType](c Config, id int) *Layer {
	return &Layer{
		LN0:     layernorm.New[T](c.DModel, 1e-6),
		LN1:     layernorm.New[T](c.DModel, 1e-6),
		LN2:     layernorm.New[T](c.DModel, 1e-6),
		ChanMix: NewChannelMix[T](c, id),
		TimeMix: NewTimeMix[T](c, id),
		ID:      id,
	}
}

func (m *Layer) ForwardSingle(x ag.Node, state *LayerState) ag.Node {
	if m.ID == 0 {
		x = m.LN0.Forward(x)[0]
	}
	x = ag.Add(x, m.TimeMix.ForwardSingle(m.LN1.Forward(x)[0], state))
	x = ag.Add(x, m.ChanMix.ForwardSingle(m.LN2.Forward(x)[0], state))
	return x
}

func (m *Layer) ForwardSequence(x []ag.Node, state *LayerState) []ag.Node {
	residual := func(other []ag.Node) []ag.Node {
		for i := range x {
			x[i] = ag.Add(x[i], other[i])
		}
		return x
	}

	if m.ID == 0 {
		x = m.LN0.Forward(x...)
	}
	x = residual(m.TimeMix.ForwardSequence(m.LN1.Forward(x...), state))
	x = residual(m.ChanMix.ForwardSequence(m.LN2.Forward(x...), state))
	return x
}
