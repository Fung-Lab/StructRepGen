#!/bin/sh

echo "=== BEHLER (TEST) ==="
python tests/test_behler.py

echo "=== EXTRACTION ==="
python examples/example_extract.py

echo "=== CVAE TRAINER ==="
python examples/example_cvae_trainer.py

echo "=== SURROGATE TRAINER ==="
python examples/example_surrogate_trainer.py

echo "=== GENERATION ==="
python examples/example_generator.py

echo "=== RECONSTRUCTION ==="
python examples/example_reconstruction.py