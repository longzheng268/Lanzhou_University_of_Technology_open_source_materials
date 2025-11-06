// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Mon Oct 28 16:04:11 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_dmem_0_0_sim_netlist.v
// Design      : risc32_dmem_0_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dmem
   (d_out,
    w_ena,
    addr,
    clk,
    d_in);
  output [31:0]d_out;
  input w_ena;
  input [7:0]addr;
  input clk;
  input [31:0]d_in;

  wire [7:0]addr;
  wire clk;
  wire [31:0]d_in;
  wire [31:0]d_out;
  wire mem_reg_0_127_0_0__0_n_0;
  wire mem_reg_0_127_0_0__10_n_0;
  wire mem_reg_0_127_0_0__11_n_0;
  wire mem_reg_0_127_0_0__12_n_0;
  wire mem_reg_0_127_0_0__13_n_0;
  wire mem_reg_0_127_0_0__14_n_0;
  wire mem_reg_0_127_0_0__15_n_0;
  wire mem_reg_0_127_0_0__16_n_0;
  wire mem_reg_0_127_0_0__17_n_0;
  wire mem_reg_0_127_0_0__18_n_0;
  wire mem_reg_0_127_0_0__19_n_0;
  wire mem_reg_0_127_0_0__1_n_0;
  wire mem_reg_0_127_0_0__20_n_0;
  wire mem_reg_0_127_0_0__21_n_0;
  wire mem_reg_0_127_0_0__22_n_0;
  wire mem_reg_0_127_0_0__23_n_0;
  wire mem_reg_0_127_0_0__24_n_0;
  wire mem_reg_0_127_0_0__25_n_0;
  wire mem_reg_0_127_0_0__26_n_0;
  wire mem_reg_0_127_0_0__27_n_0;
  wire mem_reg_0_127_0_0__28_n_0;
  wire mem_reg_0_127_0_0__29_n_0;
  wire mem_reg_0_127_0_0__2_n_0;
  wire mem_reg_0_127_0_0__30_n_0;
  wire mem_reg_0_127_0_0__3_n_0;
  wire mem_reg_0_127_0_0__4_n_0;
  wire mem_reg_0_127_0_0__5_n_0;
  wire mem_reg_0_127_0_0__6_n_0;
  wire mem_reg_0_127_0_0__7_n_0;
  wire mem_reg_0_127_0_0__8_n_0;
  wire mem_reg_0_127_0_0__9_n_0;
  wire mem_reg_0_127_0_0_i_1_n_0;
  wire mem_reg_0_127_0_0_n_0;
  wire mem_reg_0_15_0_0__0_n_0;
  wire mem_reg_0_15_0_0__10_n_0;
  wire mem_reg_0_15_0_0__11_n_0;
  wire mem_reg_0_15_0_0__12_n_0;
  wire mem_reg_0_15_0_0__13_n_0;
  wire mem_reg_0_15_0_0__14_n_0;
  wire mem_reg_0_15_0_0__15_n_0;
  wire mem_reg_0_15_0_0__16_n_0;
  wire mem_reg_0_15_0_0__17_n_0;
  wire mem_reg_0_15_0_0__18_n_0;
  wire mem_reg_0_15_0_0__19_n_0;
  wire mem_reg_0_15_0_0__1_n_0;
  wire mem_reg_0_15_0_0__20_n_0;
  wire mem_reg_0_15_0_0__21_n_0;
  wire mem_reg_0_15_0_0__22_n_0;
  wire mem_reg_0_15_0_0__23_n_0;
  wire mem_reg_0_15_0_0__24_n_0;
  wire mem_reg_0_15_0_0__25_n_0;
  wire mem_reg_0_15_0_0__26_n_0;
  wire mem_reg_0_15_0_0__27_n_0;
  wire mem_reg_0_15_0_0__28_n_0;
  wire mem_reg_0_15_0_0__29_n_0;
  wire mem_reg_0_15_0_0__2_n_0;
  wire mem_reg_0_15_0_0__30_n_0;
  wire mem_reg_0_15_0_0__3_n_0;
  wire mem_reg_0_15_0_0__4_n_0;
  wire mem_reg_0_15_0_0__5_n_0;
  wire mem_reg_0_15_0_0__6_n_0;
  wire mem_reg_0_15_0_0__7_n_0;
  wire mem_reg_0_15_0_0__8_n_0;
  wire mem_reg_0_15_0_0__9_n_0;
  wire mem_reg_0_15_0_0_i_1_n_0;
  wire mem_reg_0_15_0_0_n_0;
  wire w_ena;

  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[0]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0_n_0),
        .O(d_out[0]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[10]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__9_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__9_n_0),
        .O(d_out[10]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[11]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__10_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__10_n_0),
        .O(d_out[11]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[12]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__11_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__11_n_0),
        .O(d_out[12]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[13]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__12_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__12_n_0),
        .O(d_out[13]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[14]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__13_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__13_n_0),
        .O(d_out[14]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[15]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__14_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__14_n_0),
        .O(d_out[15]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[16]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__15_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__15_n_0),
        .O(d_out[16]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[17]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__16_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__16_n_0),
        .O(d_out[17]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[18]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__17_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__17_n_0),
        .O(d_out[18]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[19]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__18_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__18_n_0),
        .O(d_out[19]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[1]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__0_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__0_n_0),
        .O(d_out[1]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[20]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__19_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__19_n_0),
        .O(d_out[20]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[21]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__20_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__20_n_0),
        .O(d_out[21]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[22]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__21_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__21_n_0),
        .O(d_out[22]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[23]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__22_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__22_n_0),
        .O(d_out[23]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[24]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__23_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__23_n_0),
        .O(d_out[24]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[25]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__24_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__24_n_0),
        .O(d_out[25]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[26]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__25_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__25_n_0),
        .O(d_out[26]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[27]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__26_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__26_n_0),
        .O(d_out[27]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[28]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__27_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__27_n_0),
        .O(d_out[28]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[29]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__28_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__28_n_0),
        .O(d_out[29]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[2]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__1_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__1_n_0),
        .O(d_out[2]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[30]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__29_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__29_n_0),
        .O(d_out[30]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[31]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__30_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__30_n_0),
        .O(d_out[31]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[3]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__2_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__2_n_0),
        .O(d_out[3]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[4]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__3_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__3_n_0),
        .O(d_out[4]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[5]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__4_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__4_n_0),
        .O(d_out[5]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[6]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__5_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__5_n_0),
        .O(d_out[6]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[7]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__6_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__6_n_0),
        .O(d_out[7]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[8]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__7_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__7_n_0),
        .O(d_out[8]));
  LUT6 #(
    .INIT(64'h0004FFFF00040000)) 
    \d_out[9]_INST_0 
       (.I0(addr[5]),
        .I1(mem_reg_0_15_0_0__8_n_0),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .I5(mem_reg_0_127_0_0__8_n_0),
        .O(d_out[9]));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "0" *) 
  (* ram_slice_end = "0" *) 
  RAM128X1S mem_reg_0_127_0_0
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[0]),
        .O(mem_reg_0_127_0_0_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "1" *) 
  (* ram_slice_end = "1" *) 
  RAM128X1S mem_reg_0_127_0_0__0
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[1]),
        .O(mem_reg_0_127_0_0__0_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "2" *) 
  (* ram_slice_end = "2" *) 
  RAM128X1S mem_reg_0_127_0_0__1
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[2]),
        .O(mem_reg_0_127_0_0__1_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "11" *) 
  (* ram_slice_end = "11" *) 
  RAM128X1S mem_reg_0_127_0_0__10
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[11]),
        .O(mem_reg_0_127_0_0__10_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "12" *) 
  (* ram_slice_end = "12" *) 
  RAM128X1S mem_reg_0_127_0_0__11
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[12]),
        .O(mem_reg_0_127_0_0__11_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "13" *) 
  (* ram_slice_end = "13" *) 
  RAM128X1S mem_reg_0_127_0_0__12
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[13]),
        .O(mem_reg_0_127_0_0__12_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "14" *) 
  (* ram_slice_end = "14" *) 
  RAM128X1S mem_reg_0_127_0_0__13
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[14]),
        .O(mem_reg_0_127_0_0__13_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "15" *) 
  (* ram_slice_end = "15" *) 
  RAM128X1S mem_reg_0_127_0_0__14
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[15]),
        .O(mem_reg_0_127_0_0__14_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "16" *) 
  (* ram_slice_end = "16" *) 
  RAM128X1S mem_reg_0_127_0_0__15
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[16]),
        .O(mem_reg_0_127_0_0__15_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "17" *) 
  (* ram_slice_end = "17" *) 
  RAM128X1S mem_reg_0_127_0_0__16
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[17]),
        .O(mem_reg_0_127_0_0__16_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "18" *) 
  (* ram_slice_end = "18" *) 
  RAM128X1S mem_reg_0_127_0_0__17
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[18]),
        .O(mem_reg_0_127_0_0__17_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "19" *) 
  (* ram_slice_end = "19" *) 
  RAM128X1S mem_reg_0_127_0_0__18
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[19]),
        .O(mem_reg_0_127_0_0__18_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "20" *) 
  (* ram_slice_end = "20" *) 
  RAM128X1S mem_reg_0_127_0_0__19
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[20]),
        .O(mem_reg_0_127_0_0__19_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "3" *) 
  (* ram_slice_end = "3" *) 
  RAM128X1S mem_reg_0_127_0_0__2
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[3]),
        .O(mem_reg_0_127_0_0__2_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "21" *) 
  (* ram_slice_end = "21" *) 
  RAM128X1S mem_reg_0_127_0_0__20
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[21]),
        .O(mem_reg_0_127_0_0__20_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "22" *) 
  (* ram_slice_end = "22" *) 
  RAM128X1S mem_reg_0_127_0_0__21
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[22]),
        .O(mem_reg_0_127_0_0__21_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "23" *) 
  (* ram_slice_end = "23" *) 
  RAM128X1S mem_reg_0_127_0_0__22
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[23]),
        .O(mem_reg_0_127_0_0__22_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "24" *) 
  (* ram_slice_end = "24" *) 
  RAM128X1S mem_reg_0_127_0_0__23
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[24]),
        .O(mem_reg_0_127_0_0__23_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "25" *) 
  (* ram_slice_end = "25" *) 
  RAM128X1S mem_reg_0_127_0_0__24
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[25]),
        .O(mem_reg_0_127_0_0__24_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "26" *) 
  (* ram_slice_end = "26" *) 
  RAM128X1S mem_reg_0_127_0_0__25
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[26]),
        .O(mem_reg_0_127_0_0__25_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "27" *) 
  (* ram_slice_end = "27" *) 
  RAM128X1S mem_reg_0_127_0_0__26
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[27]),
        .O(mem_reg_0_127_0_0__26_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "28" *) 
  (* ram_slice_end = "28" *) 
  RAM128X1S mem_reg_0_127_0_0__27
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[28]),
        .O(mem_reg_0_127_0_0__27_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "29" *) 
  (* ram_slice_end = "29" *) 
  RAM128X1S mem_reg_0_127_0_0__28
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[29]),
        .O(mem_reg_0_127_0_0__28_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "30" *) 
  (* ram_slice_end = "30" *) 
  RAM128X1S mem_reg_0_127_0_0__29
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[30]),
        .O(mem_reg_0_127_0_0__29_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "4" *) 
  (* ram_slice_end = "4" *) 
  RAM128X1S mem_reg_0_127_0_0__3
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[4]),
        .O(mem_reg_0_127_0_0__3_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "31" *) 
  (* ram_slice_end = "31" *) 
  RAM128X1S mem_reg_0_127_0_0__30
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[31]),
        .O(mem_reg_0_127_0_0__30_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "5" *) 
  (* ram_slice_end = "5" *) 
  RAM128X1S mem_reg_0_127_0_0__4
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[5]),
        .O(mem_reg_0_127_0_0__4_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "6" *) 
  (* ram_slice_end = "6" *) 
  RAM128X1S mem_reg_0_127_0_0__5
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[6]),
        .O(mem_reg_0_127_0_0__5_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "7" *) 
  (* ram_slice_end = "7" *) 
  RAM128X1S mem_reg_0_127_0_0__6
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[7]),
        .O(mem_reg_0_127_0_0__6_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "8" *) 
  (* ram_slice_end = "8" *) 
  RAM128X1S mem_reg_0_127_0_0__7
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[8]),
        .O(mem_reg_0_127_0_0__7_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "9" *) 
  (* ram_slice_end = "9" *) 
  RAM128X1S mem_reg_0_127_0_0__8
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[9]),
        .O(mem_reg_0_127_0_0__8_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "127" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "10" *) 
  (* ram_slice_end = "10" *) 
  RAM128X1S mem_reg_0_127_0_0__9
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(addr[4]),
        .A5(addr[5]),
        .A6(addr[6]),
        .D(d_in[10]),
        .O(mem_reg_0_127_0_0__9_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_127_0_0_i_1_n_0));
  LUT2 #(
    .INIT(4'h2)) 
    mem_reg_0_127_0_0_i_1
       (.I0(w_ena),
        .I1(addr[7]),
        .O(mem_reg_0_127_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "0" *) 
  (* ram_slice_end = "0" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[0]),
        .O(mem_reg_0_15_0_0_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "1" *) 
  (* ram_slice_end = "1" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__0
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[1]),
        .O(mem_reg_0_15_0_0__0_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "2" *) 
  (* ram_slice_end = "2" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__1
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[2]),
        .O(mem_reg_0_15_0_0__1_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "11" *) 
  (* ram_slice_end = "11" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__10
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[11]),
        .O(mem_reg_0_15_0_0__10_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "12" *) 
  (* ram_slice_end = "12" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__11
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[12]),
        .O(mem_reg_0_15_0_0__11_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "13" *) 
  (* ram_slice_end = "13" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__12
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[13]),
        .O(mem_reg_0_15_0_0__12_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "14" *) 
  (* ram_slice_end = "14" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__13
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[14]),
        .O(mem_reg_0_15_0_0__13_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "15" *) 
  (* ram_slice_end = "15" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__14
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[15]),
        .O(mem_reg_0_15_0_0__14_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "16" *) 
  (* ram_slice_end = "16" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__15
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[16]),
        .O(mem_reg_0_15_0_0__15_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "17" *) 
  (* ram_slice_end = "17" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__16
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[17]),
        .O(mem_reg_0_15_0_0__16_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "18" *) 
  (* ram_slice_end = "18" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__17
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[18]),
        .O(mem_reg_0_15_0_0__17_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "19" *) 
  (* ram_slice_end = "19" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__18
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[19]),
        .O(mem_reg_0_15_0_0__18_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "20" *) 
  (* ram_slice_end = "20" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__19
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[20]),
        .O(mem_reg_0_15_0_0__19_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "3" *) 
  (* ram_slice_end = "3" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__2
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[3]),
        .O(mem_reg_0_15_0_0__2_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "21" *) 
  (* ram_slice_end = "21" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__20
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[21]),
        .O(mem_reg_0_15_0_0__20_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "22" *) 
  (* ram_slice_end = "22" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__21
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[22]),
        .O(mem_reg_0_15_0_0__21_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "23" *) 
  (* ram_slice_end = "23" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__22
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[23]),
        .O(mem_reg_0_15_0_0__22_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "24" *) 
  (* ram_slice_end = "24" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__23
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[24]),
        .O(mem_reg_0_15_0_0__23_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "25" *) 
  (* ram_slice_end = "25" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__24
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[25]),
        .O(mem_reg_0_15_0_0__24_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "26" *) 
  (* ram_slice_end = "26" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__25
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[26]),
        .O(mem_reg_0_15_0_0__25_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "27" *) 
  (* ram_slice_end = "27" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__26
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[27]),
        .O(mem_reg_0_15_0_0__26_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "28" *) 
  (* ram_slice_end = "28" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__27
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[28]),
        .O(mem_reg_0_15_0_0__27_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "29" *) 
  (* ram_slice_end = "29" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__28
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[29]),
        .O(mem_reg_0_15_0_0__28_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "30" *) 
  (* ram_slice_end = "30" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__29
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[30]),
        .O(mem_reg_0_15_0_0__29_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "4" *) 
  (* ram_slice_end = "4" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__3
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[4]),
        .O(mem_reg_0_15_0_0__3_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "31" *) 
  (* ram_slice_end = "31" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__30
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[31]),
        .O(mem_reg_0_15_0_0__30_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "5" *) 
  (* ram_slice_end = "5" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__4
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[5]),
        .O(mem_reg_0_15_0_0__4_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "6" *) 
  (* ram_slice_end = "6" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__5
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[6]),
        .O(mem_reg_0_15_0_0__5_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "7" *) 
  (* ram_slice_end = "7" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__6
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[7]),
        .O(mem_reg_0_15_0_0__6_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "8" *) 
  (* ram_slice_end = "8" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__7
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[8]),
        .O(mem_reg_0_15_0_0__7_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "9" *) 
  (* ram_slice_end = "9" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__8
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[9]),
        .O(mem_reg_0_15_0_0__8_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  (* RTL_RAM_BITS = "4128" *) 
  (* RTL_RAM_NAME = "inst/mem_reg" *) 
  (* RTL_RAM_TYPE = "RAM_SP" *) 
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  (* XILINX_TRANSFORM_PINMAP = "GND:A4" *) 
  (* ram_addr_begin = "128" *) 
  (* ram_addr_end = "128" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "10" *) 
  (* ram_slice_end = "10" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    mem_reg_0_15_0_0__9
       (.A0(addr[0]),
        .A1(addr[1]),
        .A2(addr[2]),
        .A3(addr[3]),
        .A4(1'b0),
        .D(d_in[10]),
        .O(mem_reg_0_15_0_0__9_n_0),
        .WCLK(clk),
        .WE(mem_reg_0_15_0_0_i_1_n_0));
  LUT5 #(
    .INIT(32'h00020000)) 
    mem_reg_0_15_0_0_i_1
       (.I0(w_ena),
        .I1(addr[5]),
        .I2(addr[4]),
        .I3(addr[6]),
        .I4(addr[7]),
        .O(mem_reg_0_15_0_0_i_1_n_0));
endmodule

(* CHECK_LICENSE_TYPE = "risc32_dmem_0_0,dmem,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "dmem,Vivado 2022.2" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (clk,
    w_ena,
    addr,
    d_in,
    d_out);
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk, FREQ_HZ 100000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN risc32_clk_0, INSERT_VIP 0" *) input clk;
  input w_ena;
  input [31:0]addr;
  input [31:0]d_in;
  output [31:0]d_out;

  wire [31:0]addr;
  wire clk;
  wire [31:0]d_in;
  wire [31:0]d_out;
  wire w_ena;

  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dmem inst
       (.addr(addr[7:0]),
        .clk(clk),
        .d_in(d_in),
        .d_out(d_out),
        .w_ena(w_ena));
endmodule
`ifndef GLBL
`define GLBL
`timescale  1 ps / 1 ps

module glbl ();

    parameter ROC_WIDTH = 100000;
    parameter TOC_WIDTH = 0;
    parameter GRES_WIDTH = 10000;
    parameter GRES_START = 10000;

//--------   STARTUP Globals --------------
    wire GSR;
    wire GTS;
    wire GWE;
    wire PRLD;
    wire GRESTORE;
    tri1 p_up_tmp;
    tri (weak1, strong0) PLL_LOCKG = p_up_tmp;

    wire PROGB_GLBL;
    wire CCLKO_GLBL;
    wire FCSBO_GLBL;
    wire [3:0] DO_GLBL;
    wire [3:0] DI_GLBL;
   
    reg GSR_int;
    reg GTS_int;
    reg PRLD_int;
    reg GRESTORE_int;

//--------   JTAG Globals --------------
    wire JTAG_TDO_GLBL;
    wire JTAG_TCK_GLBL;
    wire JTAG_TDI_GLBL;
    wire JTAG_TMS_GLBL;
    wire JTAG_TRST_GLBL;

    reg JTAG_CAPTURE_GLBL;
    reg JTAG_RESET_GLBL;
    reg JTAG_SHIFT_GLBL;
    reg JTAG_UPDATE_GLBL;
    reg JTAG_RUNTEST_GLBL;

    reg JTAG_SEL1_GLBL = 0;
    reg JTAG_SEL2_GLBL = 0 ;
    reg JTAG_SEL3_GLBL = 0;
    reg JTAG_SEL4_GLBL = 0;

    reg JTAG_USER_TDO1_GLBL = 1'bz;
    reg JTAG_USER_TDO2_GLBL = 1'bz;
    reg JTAG_USER_TDO3_GLBL = 1'bz;
    reg JTAG_USER_TDO4_GLBL = 1'bz;

    assign (strong1, weak0) GSR = GSR_int;
    assign (strong1, weak0) GTS = GTS_int;
    assign (weak1, weak0) PRLD = PRLD_int;
    assign (strong1, weak0) GRESTORE = GRESTORE_int;

    initial begin
	GSR_int = 1'b1;
	PRLD_int = 1'b1;
	#(ROC_WIDTH)
	GSR_int = 1'b0;
	PRLD_int = 1'b0;
    end

    initial begin
	GTS_int = 1'b1;
	#(TOC_WIDTH)
	GTS_int = 1'b0;
    end

    initial begin 
	GRESTORE_int = 1'b0;
	#(GRES_START);
	GRESTORE_int = 1'b1;
	#(GRES_WIDTH);
	GRESTORE_int = 1'b0;
    end

endmodule
`endif
