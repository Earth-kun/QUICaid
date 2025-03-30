#!/bin/bash

parser="~/QUIC/live_program/quicaid.sh 1 0 10.10.3.10 out1.csv"

parallel ::: "$parser" "~/QUIC/Data_Collection/benign_new.sh"
