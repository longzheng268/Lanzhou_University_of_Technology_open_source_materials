// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Tue Oct 29 10:56:54 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode synth_stub
//               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_segment_0_0/risc32_segment_0_0_stub.v
// Design      : risc32_segment_0_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "segment,Vivado 2022.2" *)
module risc32_segment_0_0(clk, rst_n, Data_i, AN, seg_data_o)
/* synthesis syn_black_box black_box_pad_pin="clk,rst_n,Data_i[31:0],AN[7:0],seg_data_o[7:0]" */;
  input clk;
  input rst_n;
  input [31:0]Data_i;
  output [7:0]AN;
  output [7:0]seg_data_o;
endmodule
