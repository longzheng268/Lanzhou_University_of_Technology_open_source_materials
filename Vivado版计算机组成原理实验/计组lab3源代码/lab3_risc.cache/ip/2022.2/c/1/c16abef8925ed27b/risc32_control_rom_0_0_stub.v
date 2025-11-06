// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Mon Oct 28 16:04:11 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode synth_stub -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_control_rom_0_0_stub.v
// Design      : risc32_control_rom_0_0
// Purpose     : Stub declaration of top-level module interface
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "control_rom,Vivado 2022.2" *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix(inst, imm_sel, bsel, mem_sel, alusel, w_ena, mem_ena)
/* synthesis syn_black_box black_box_pad_pin="inst[8:0],imm_sel,bsel,mem_sel,alusel[2:0],w_ena,mem_ena" */;
  input [8:0]inst;
  output imm_sel;
  output bsel;
  output mem_sel;
  output [2:0]alusel;
  output w_ena;
  output mem_ena;
endmodule
