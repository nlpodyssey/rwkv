// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkv

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

// Model implements the RWKV Language Modeling task.
type Model struct {
	nn.Module
	Layers []*Layer
	Config Config
}

// Config is the configuration of the RWKV model.
type Config struct {
	DModel       int
	NumLayers    int
	RescaleLayer int
}

func init() {
	gob.Register(&Model{})
}

// New returns a new RWKV model.
func New[T float.DType](c Config) *Model {
	m := &Model{Config: c}
	for i := 0; i < c.NumLayers; i++ {
		m.Layers = append(m.Layers, NewLayer[T](c, i))
	}
	return m
}

func (m *Model) Forward(x ag.Node, state State) (ag.Node, State) {
	if len(state) == 0 {
		state = NewState(m.Config)
	}
	for i, layer := range m.Layers {
		x = layer.Forward(x, state[i])

		if (i+1)%m.Config.RescaleLayer == 0 {
			x = ag.ProdScalar(x, ag.Scalar(0.5))
		}
	}
	return x, state
}
