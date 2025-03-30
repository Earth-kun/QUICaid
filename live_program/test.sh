#!/bin/bash

parser="~/QUIC/live_program/quicaid.sh 1 0 10.10.3.10 out2.csv"

parallel ::: "$parser" "~/QUIC/Data_Collection/benign_new.sh"
