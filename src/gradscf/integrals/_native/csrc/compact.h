// Internal dispatch functions; the public FFI ABI remains unchanged.
#pragma once
#include "tables.h"

void packed_eri(IntegralTables& t, int layout, int cart, double* out, size_t output_size);
void density_fitting(IntegralTables& t, int layout, int cart, int split, double* out, size_t output_size);
