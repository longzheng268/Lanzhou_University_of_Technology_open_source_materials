//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
//Date        : Wed Nov  5 17:52:47 2025
//Host        : DESKTOP-5VI2UC6 running 64-bit major release  (build 9200)
//Command     : generate_target risc32_wrapper.bd
//Design      : risc32_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module risc32_wrapper
   (SSEG_AN,
    SSEG_CA,
    alu_out_0,
    clk,
    clk_in1_0,
    pc_clr);
  output [7:0]SSEG_AN;
  output [7:0]SSEG_CA;
  output [31:0]alu_out_0;
  input clk;
  input clk_in1_0;
  input pc_clr;

  wire [7:0]SSEG_AN;
  wire [7:0]SSEG_CA;
  wire [31:0]alu_out_0;
  wire clk;
  wire clk_in1_0;
  wire pc_clr;

  risc32 risc32_i
       (.SSEG_AN(SSEG_AN),
        .SSEG_CA(SSEG_CA),
        .alu_out_0(alu_out_0),
        .clk(clk),
        .clk_in1_0(clk_in1_0),
        .pc_clr(pc_clr));
endmodule
