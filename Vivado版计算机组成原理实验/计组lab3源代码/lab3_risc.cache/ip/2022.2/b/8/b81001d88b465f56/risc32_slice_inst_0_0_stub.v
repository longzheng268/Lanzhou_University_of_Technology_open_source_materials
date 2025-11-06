// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Mon Oct 28 16:04:12 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode synth_stub -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_slice_inst_0_0_stub.v
// Design      : risc32_slice_inst_0_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "slice_inst,Vivado 2022.2" *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix(instruction, inst_31, inst_30, inst_31_25, 
  inst_24_20, inst_19_15, inst_14_12, inst_11_7, inst_6_2)
/* synthesis syn_black_box black_box_pad_pin="instruction[31:0],inst_31,inst_30,inst_31_25[6:0],inst_24_20[4:0],inst_19_15[4:0],inst_14_12[2:0],inst_11_7[4:0],inst_6_2[4:0]" */;
  input [31:0]instruction;
  output inst_31;
  output inst_30;
  output [6:0]inst_31_25;
  output [4:0]inst_24_20;
  output [4:0]inst_19_15;
  output [2:0]inst_14_12;
  output [4:0]inst_11_7;
  output [4:0]inst_6_2;
endmodule
