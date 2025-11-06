// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Mon Oct 28 16:04:11 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_regfile_0_0_sim_netlist.v
// Design      : risc32_regfile_0_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_regfile
   (reg_data1,
    reg_data2,
    clk,
    w_data,
    w_ena,
    w_addr,
    r_addr1,
    r_addr2);
  output [31:0]reg_data1;
  output [31:0]reg_data2;
  input clk;
  input [31:0]w_data;
  input w_ena;
  input [4:0]w_addr;
  input [4:0]r_addr1;
  input [4:0]r_addr2;

  wire clk;
  wire [4:0]r_addr1;
  wire [4:0]r_addr2;
  wire [31:0]reg_data1;
  wire [31:0]reg_data10;
  wire [31:0]reg_data2;
  wire [31:0]reg_data20;
  wire [4:0]w_addr;
  wire [31:0]w_data;
  wire w_ena;
  wire [1:0]NLW_reg_array_reg_r1_0_31_0_5_DOD_UNCONNECTED;
  wire [1:0]NLW_reg_array_reg_r1_0_31_12_17_DOD_UNCONNECTED;
  wire [1:0]NLW_reg_array_reg_r1_0_31_18_23_DOD_UNCONNECTED;
  wire [1:0]NLW_reg_array_reg_r1_0_31_24_29_DOD_UNCONNECTED;
  wire NLW_reg_array_reg_r1_0_31_30_31_SPO_UNCONNECTED;
  wire NLW_reg_array_reg_r1_0_31_30_31__0_SPO_UNCONNECTED;
  wire [1:0]NLW_reg_array_reg_r1_0_31_6_11_DOD_UNCONNECTED;
  wire [1:0]NLW_reg_array_reg_r2_0_31_0_5_DOD_UNCONNECTED;
  wire [1:0]NLW_reg_array_reg_r2_0_31_12_17_DOD_UNCONNECTED;
  wire [1:0]NLW_reg_array_reg_r2_0_31_18_23_DOD_UNCONNECTED;
  wire [1:0]NLW_reg_array_reg_r2_0_31_24_29_DOD_UNCONNECTED;
  wire NLW_reg_array_reg_r2_0_31_30_31_SPO_UNCONNECTED;
  wire NLW_reg_array_reg_r2_0_31_30_31__0_SPO_UNCONNECTED;
  wire [1:0]NLW_reg_array_reg_r2_0_31_6_11_DOD_UNCONNECTED;

  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r1_0_31_0_5" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "0" *) 
  (* ram_slice_end = "5" *) 
  RAM32M reg_array_reg_r1_0_31_0_5
       (.ADDRA(r_addr1),
        .ADDRB(r_addr1),
        .ADDRC(r_addr1),
        .ADDRD(w_addr),
        .DIA(w_data[1:0]),
        .DIB(w_data[3:2]),
        .DIC(w_data[5:4]),
        .DID({1'b0,1'b0}),
        .DOA(reg_data10[1:0]),
        .DOB(reg_data10[3:2]),
        .DOC(reg_data10[5:4]),
        .DOD(NLW_reg_array_reg_r1_0_31_0_5_DOD_UNCONNECTED[1:0]),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r1_0_31_12_17" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "12" *) 
  (* ram_slice_end = "17" *) 
  RAM32M reg_array_reg_r1_0_31_12_17
       (.ADDRA(r_addr1),
        .ADDRB(r_addr1),
        .ADDRC(r_addr1),
        .ADDRD(w_addr),
        .DIA(w_data[13:12]),
        .DIB(w_data[15:14]),
        .DIC(w_data[17:16]),
        .DID({1'b0,1'b0}),
        .DOA(reg_data10[13:12]),
        .DOB(reg_data10[15:14]),
        .DOC(reg_data10[17:16]),
        .DOD(NLW_reg_array_reg_r1_0_31_12_17_DOD_UNCONNECTED[1:0]),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r1_0_31_18_23" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "18" *) 
  (* ram_slice_end = "23" *) 
  RAM32M reg_array_reg_r1_0_31_18_23
       (.ADDRA(r_addr1),
        .ADDRB(r_addr1),
        .ADDRC(r_addr1),
        .ADDRD(w_addr),
        .DIA(w_data[19:18]),
        .DIB(w_data[21:20]),
        .DIC(w_data[23:22]),
        .DID({1'b0,1'b0}),
        .DOA(reg_data10[19:18]),
        .DOB(reg_data10[21:20]),
        .DOC(reg_data10[23:22]),
        .DOD(NLW_reg_array_reg_r1_0_31_18_23_DOD_UNCONNECTED[1:0]),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r1_0_31_24_29" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "24" *) 
  (* ram_slice_end = "29" *) 
  RAM32M reg_array_reg_r1_0_31_24_29
       (.ADDRA(r_addr1),
        .ADDRB(r_addr1),
        .ADDRC(r_addr1),
        .ADDRD(w_addr),
        .DIA(w_data[25:24]),
        .DIB(w_data[27:26]),
        .DIC(w_data[29:28]),
        .DID({1'b0,1'b0}),
        .DOA(reg_data10[25:24]),
        .DOB(reg_data10[27:26]),
        .DOC(reg_data10[29:28]),
        .DOD(NLW_reg_array_reg_r1_0_31_24_29_DOD_UNCONNECTED[1:0]),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r1_0_31_30_31" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "30" *) 
  (* ram_slice_end = "31" *) 
  RAM32X1D reg_array_reg_r1_0_31_30_31
       (.A0(w_addr[0]),
        .A1(w_addr[1]),
        .A2(w_addr[2]),
        .A3(w_addr[3]),
        .A4(w_addr[4]),
        .D(w_data[30]),
        .DPO(reg_data10[30]),
        .DPRA0(r_addr1[0]),
        .DPRA1(r_addr1[1]),
        .DPRA2(r_addr1[2]),
        .DPRA3(r_addr1[3]),
        .DPRA4(r_addr1[4]),
        .SPO(NLW_reg_array_reg_r1_0_31_30_31_SPO_UNCONNECTED),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r1_0_31_30_31" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "30" *) 
  (* ram_slice_end = "31" *) 
  RAM32X1D reg_array_reg_r1_0_31_30_31__0
       (.A0(w_addr[0]),
        .A1(w_addr[1]),
        .A2(w_addr[2]),
        .A3(w_addr[3]),
        .A4(w_addr[4]),
        .D(w_data[31]),
        .DPO(reg_data10[31]),
        .DPRA0(r_addr1[0]),
        .DPRA1(r_addr1[1]),
        .DPRA2(r_addr1[2]),
        .DPRA3(r_addr1[3]),
        .DPRA4(r_addr1[4]),
        .SPO(NLW_reg_array_reg_r1_0_31_30_31__0_SPO_UNCONNECTED),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r1_0_31_6_11" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "6" *) 
  (* ram_slice_end = "11" *) 
  RAM32M reg_array_reg_r1_0_31_6_11
       (.ADDRA(r_addr1),
        .ADDRB(r_addr1),
        .ADDRC(r_addr1),
        .ADDRD(w_addr),
        .DIA(w_data[7:6]),
        .DIB(w_data[9:8]),
        .DIC(w_data[11:10]),
        .DID({1'b0,1'b0}),
        .DOA(reg_data10[7:6]),
        .DOB(reg_data10[9:8]),
        .DOC(reg_data10[11:10]),
        .DOD(NLW_reg_array_reg_r1_0_31_6_11_DOD_UNCONNECTED[1:0]),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r2_0_31_0_5" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "0" *) 
  (* ram_slice_end = "5" *) 
  RAM32M reg_array_reg_r2_0_31_0_5
       (.ADDRA(r_addr2),
        .ADDRB(r_addr2),
        .ADDRC(r_addr2),
        .ADDRD(w_addr),
        .DIA(w_data[1:0]),
        .DIB(w_data[3:2]),
        .DIC(w_data[5:4]),
        .DID({1'b0,1'b0}),
        .DOA(reg_data20[1:0]),
        .DOB(reg_data20[3:2]),
        .DOC(reg_data20[5:4]),
        .DOD(NLW_reg_array_reg_r2_0_31_0_5_DOD_UNCONNECTED[1:0]),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r2_0_31_12_17" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "12" *) 
  (* ram_slice_end = "17" *) 
  RAM32M reg_array_reg_r2_0_31_12_17
       (.ADDRA(r_addr2),
        .ADDRB(r_addr2),
        .ADDRC(r_addr2),
        .ADDRD(w_addr),
        .DIA(w_data[13:12]),
        .DIB(w_data[15:14]),
        .DIC(w_data[17:16]),
        .DID({1'b0,1'b0}),
        .DOA(reg_data20[13:12]),
        .DOB(reg_data20[15:14]),
        .DOC(reg_data20[17:16]),
        .DOD(NLW_reg_array_reg_r2_0_31_12_17_DOD_UNCONNECTED[1:0]),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r2_0_31_18_23" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "18" *) 
  (* ram_slice_end = "23" *) 
  RAM32M reg_array_reg_r2_0_31_18_23
       (.ADDRA(r_addr2),
        .ADDRB(r_addr2),
        .ADDRC(r_addr2),
        .ADDRD(w_addr),
        .DIA(w_data[19:18]),
        .DIB(w_data[21:20]),
        .DIC(w_data[23:22]),
        .DID({1'b0,1'b0}),
        .DOA(reg_data20[19:18]),
        .DOB(reg_data20[21:20]),
        .DOC(reg_data20[23:22]),
        .DOD(NLW_reg_array_reg_r2_0_31_18_23_DOD_UNCONNECTED[1:0]),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r2_0_31_24_29" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "24" *) 
  (* ram_slice_end = "29" *) 
  RAM32M reg_array_reg_r2_0_31_24_29
       (.ADDRA(r_addr2),
        .ADDRB(r_addr2),
        .ADDRC(r_addr2),
        .ADDRD(w_addr),
        .DIA(w_data[25:24]),
        .DIB(w_data[27:26]),
        .DIC(w_data[29:28]),
        .DID({1'b0,1'b0}),
        .DOA(reg_data20[25:24]),
        .DOB(reg_data20[27:26]),
        .DOC(reg_data20[29:28]),
        .DOD(NLW_reg_array_reg_r2_0_31_24_29_DOD_UNCONNECTED[1:0]),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r2_0_31_30_31" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "30" *) 
  (* ram_slice_end = "31" *) 
  RAM32X1D reg_array_reg_r2_0_31_30_31
       (.A0(w_addr[0]),
        .A1(w_addr[1]),
        .A2(w_addr[2]),
        .A3(w_addr[3]),
        .A4(w_addr[4]),
        .D(w_data[30]),
        .DPO(reg_data20[30]),
        .DPRA0(r_addr2[0]),
        .DPRA1(r_addr2[1]),
        .DPRA2(r_addr2[2]),
        .DPRA3(r_addr2[3]),
        .DPRA4(r_addr2[4]),
        .SPO(NLW_reg_array_reg_r2_0_31_30_31_SPO_UNCONNECTED),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r2_0_31_30_31" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "30" *) 
  (* ram_slice_end = "31" *) 
  RAM32X1D reg_array_reg_r2_0_31_30_31__0
       (.A0(w_addr[0]),
        .A1(w_addr[1]),
        .A2(w_addr[2]),
        .A3(w_addr[3]),
        .A4(w_addr[4]),
        .D(w_data[31]),
        .DPO(reg_data20[31]),
        .DPRA0(r_addr2[0]),
        .DPRA1(r_addr2[1]),
        .DPRA2(r_addr2[2]),
        .DPRA3(r_addr2[3]),
        .DPRA4(r_addr2[4]),
        .SPO(NLW_reg_array_reg_r2_0_31_30_31__0_SPO_UNCONNECTED),
        .WCLK(clk),
        .WE(w_ena));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* RTL_RAM_BITS = "1024" *) 
  (* RTL_RAM_NAME = "inst/reg_array_reg_r2_0_31_6_11" *) 
  (* RTL_RAM_TYPE = "RAM_SDP" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "31" *) 
  (* ram_offset = "0" *) 
  (* ram_slice_begin = "6" *) 
  (* ram_slice_end = "11" *) 
  RAM32M reg_array_reg_r2_0_31_6_11
       (.ADDRA(r_addr2),
        .ADDRB(r_addr2),
        .ADDRC(r_addr2),
        .ADDRD(w_addr),
        .DIA(w_data[7:6]),
        .DIB(w_data[9:8]),
        .DIC(w_data[11:10]),
        .DID({1'b0,1'b0}),
        .DOA(reg_data20[7:6]),
        .DOB(reg_data20[9:8]),
        .DOC(reg_data20[11:10]),
        .DOD(NLW_reg_array_reg_r2_0_31_6_11_DOD_UNCONNECTED[1:0]),
        .WCLK(clk),
        .WE(w_ena));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[0]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[0]),
        .O(reg_data1[0]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[10]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[10]),
        .O(reg_data1[10]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[11]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[11]),
        .O(reg_data1[11]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[12]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[12]),
        .O(reg_data1[12]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[13]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[13]),
        .O(reg_data1[13]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[14]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[14]),
        .O(reg_data1[14]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[15]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[15]),
        .O(reg_data1[15]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[16]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[16]),
        .O(reg_data1[16]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[17]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[17]),
        .O(reg_data1[17]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[18]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[18]),
        .O(reg_data1[18]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[19]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[19]),
        .O(reg_data1[19]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[1]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[1]),
        .O(reg_data1[1]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[20]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[20]),
        .O(reg_data1[20]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[21]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[21]),
        .O(reg_data1[21]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[22]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[22]),
        .O(reg_data1[22]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[23]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[23]),
        .O(reg_data1[23]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[24]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[24]),
        .O(reg_data1[24]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[25]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[25]),
        .O(reg_data1[25]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[26]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[26]),
        .O(reg_data1[26]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[27]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[27]),
        .O(reg_data1[27]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[28]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[28]),
        .O(reg_data1[28]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[29]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[29]),
        .O(reg_data1[29]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[2]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[2]),
        .O(reg_data1[2]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[30]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[30]),
        .O(reg_data1[30]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[31]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[31]),
        .O(reg_data1[31]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[3]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[3]),
        .O(reg_data1[3]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[4]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[4]),
        .O(reg_data1[4]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[5]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[5]),
        .O(reg_data1[5]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[6]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[6]),
        .O(reg_data1[6]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[7]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[7]),
        .O(reg_data1[7]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[8]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[8]),
        .O(reg_data1[8]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data1[9]_INST_0 
       (.I0(r_addr1[4]),
        .I1(r_addr1[3]),
        .I2(r_addr1[1]),
        .I3(r_addr1[0]),
        .I4(r_addr1[2]),
        .I5(reg_data10[9]),
        .O(reg_data1[9]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[0]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[0]),
        .O(reg_data2[0]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[10]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[10]),
        .O(reg_data2[10]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[11]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[11]),
        .O(reg_data2[11]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[12]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[12]),
        .O(reg_data2[12]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[13]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[13]),
        .O(reg_data2[13]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[14]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[14]),
        .O(reg_data2[14]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[15]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[15]),
        .O(reg_data2[15]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[16]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[16]),
        .O(reg_data2[16]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[17]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[17]),
        .O(reg_data2[17]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[18]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[18]),
        .O(reg_data2[18]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[19]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[19]),
        .O(reg_data2[19]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[1]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[1]),
        .O(reg_data2[1]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[20]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[20]),
        .O(reg_data2[20]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[21]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[21]),
        .O(reg_data2[21]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[22]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[22]),
        .O(reg_data2[22]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[23]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[23]),
        .O(reg_data2[23]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[24]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[24]),
        .O(reg_data2[24]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[25]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[25]),
        .O(reg_data2[25]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[26]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[26]),
        .O(reg_data2[26]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[27]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[27]),
        .O(reg_data2[27]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[28]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[28]),
        .O(reg_data2[28]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[29]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[29]),
        .O(reg_data2[29]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[2]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[2]),
        .O(reg_data2[2]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[30]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[30]),
        .O(reg_data2[30]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[31]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[31]),
        .O(reg_data2[31]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[3]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[3]),
        .O(reg_data2[3]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[4]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[4]),
        .O(reg_data2[4]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[5]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[5]),
        .O(reg_data2[5]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[6]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[6]),
        .O(reg_data2[6]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[7]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[7]),
        .O(reg_data2[7]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[8]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[8]),
        .O(reg_data2[8]));
  LUT6 #(
    .INIT(64'hFFFFFFFE00000000)) 
    \reg_data2[9]_INST_0 
       (.I0(r_addr2[4]),
        .I1(r_addr2[3]),
        .I2(r_addr2[1]),
        .I3(r_addr2[0]),
        .I4(r_addr2[2]),
        .I5(reg_data20[9]),
        .O(reg_data2[9]));
endmodule

(* CHECK_LICENSE_TYPE = "risc32_regfile_0_0,regfile,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "regfile,Vivado 2022.2" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (clk,
    w_ena,
    r_addr1,
    r_addr2,
    w_addr,
    w_data,
    reg_data1,
    reg_data2);
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk, FREQ_HZ 100000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN risc32_clk_0, INSERT_VIP 0" *) input clk;
  input w_ena;
  input [4:0]r_addr1;
  input [4:0]r_addr2;
  input [4:0]w_addr;
  input [31:0]w_data;
  output [31:0]reg_data1;
  output [31:0]reg_data2;

  wire clk;
  wire [4:0]r_addr1;
  wire [4:0]r_addr2;
  wire [31:0]reg_data1;
  wire [31:0]reg_data2;
  wire [4:0]w_addr;
  wire [31:0]w_data;
  wire w_ena;

  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_regfile inst
       (.clk(clk),
        .r_addr1(r_addr1),
        .r_addr2(r_addr2),
        .reg_data1(reg_data1),
        .reg_data2(reg_data2),
        .w_addr(w_addr),
        .w_data(w_data),
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
