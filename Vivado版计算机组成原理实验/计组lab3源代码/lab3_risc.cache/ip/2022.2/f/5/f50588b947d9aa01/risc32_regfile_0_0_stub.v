// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Mon Oct 28 16:04:11 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode synth_stub -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_regfile_0_0_stub.v
// Design      : risc32_regfile_0_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "regfile,Vivado 2022.2" *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix(clk, w_ena, r_addr1, r_addr2, w_addr, w_data, 
  reg_data1, reg_data2)
/* synthesis syn_black_box black_box_pad_pin="clk,w_ena,r_addr1[4:0],r_addr2[4:0],w_addr[4:0],w_data[31:0],reg_data1[31:0],reg_data2[31:0]" */;
  input clk;
  input w_ena;
  input [4:0]r_addr1;
  input [4:0]r_addr2;
  input [4:0]w_addr;
  input [31:0]w_data;
  output [31:0]reg_data1;
  output [31:0]reg_data2;
endmodule
