#!/bin/bash -ex

cd tests
python -m unittest test_backend.Tf2OnnxBackendTests.test_div
