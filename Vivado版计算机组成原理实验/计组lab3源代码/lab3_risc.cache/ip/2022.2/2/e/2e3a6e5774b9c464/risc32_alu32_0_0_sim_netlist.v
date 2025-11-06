// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Mon Oct 28 16:04:54 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_alu32_0_0_sim_netlist.v
// Design      : risc32_alu32_0_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_alu32
   (alu_out,
    alu_a,
    alu_b,
    alu_out_31_sp_1,
    alu_sel,
    \alu_out[31]_0 ,
    \alu_out[31]_1 ,
    \alu_out[31]_2 ,
    alu_out_1_sp_1,
    \alu_out[1]_0 ,
    \alu_out[1]_1 ,
    alu_out_2_sp_1,
    \alu_out[2]_0 ,
    alu_out_17_sp_1,
    \alu_out[17]_0 ,
    \alu_out[17]_1 ,
    alu_out_18_sp_1,
    \alu_out[18]_0 ,
    alu_out_15_sp_1,
    \alu_out[15]_0 ,
    \alu_out[15]_1 ,
    alu_out_16_sp_1,
    alu_out_13_sp_1,
    \alu_out[13]_0 ,
    \alu_out[13]_1 ,
    alu_out_14_sp_1,
    alu_out_11_sp_1,
    \alu_out[11]_0 ,
    \alu_out[11]_1 ,
    alu_out_12_sp_1,
    alu_out_4_sp_1,
    \alu_out[4]_0 ,
    \alu_out[4]_1 ,
    alu_out_5_sp_1,
    \alu_out[5]_0 ,
    alu_out_6_sp_1,
    \alu_out[6]_0 ,
    alu_out_7_sp_1,
    \alu_out[7]_0 ,
    alu_out_8_sp_1,
    \alu_out[8]_0 ,
    alu_out_9_sp_1,
    \alu_out[9]_0 ,
    alu_out_10_sp_1,
    alu_out_19_sp_1,
    \alu_out[19]_0 ,
    alu_out_20_sp_1,
    \alu_out[20]_0 ,
    alu_out_21_sp_1,
    \alu_out[21]_0 ,
    alu_out_22_sp_1,
    \alu_out[22]_0 ,
    alu_out_23_sp_1,
    \alu_out[23]_0 ,
    alu_out_24_sp_1,
    \alu_out[24]_0 ,
    alu_out_25_sp_1,
    \alu_out[25]_0 ,
    alu_out_26_sp_1,
    \alu_out[26]_0 ,
    alu_out_27_sp_1,
    \alu_out[27]_0 ,
    alu_out_28_sp_1,
    \alu_out[28]_0 ,
    alu_out_29_sp_1,
    \alu_out[29]_0 ,
    alu_out_30_sp_1,
    alu_out_0_sp_1,
    alu_out_3_sp_1);
  output [31:0]alu_out;
  input [31:0]alu_a;
  input [31:0]alu_b;
  input alu_out_31_sp_1;
  input [2:0]alu_sel;
  input \alu_out[31]_0 ;
  input \alu_out[31]_1 ;
  input \alu_out[31]_2 ;
  input alu_out_1_sp_1;
  input \alu_out[1]_0 ;
  input \alu_out[1]_1 ;
  input alu_out_2_sp_1;
  input \alu_out[2]_0 ;
  input alu_out_17_sp_1;
  input \alu_out[17]_0 ;
  input \alu_out[17]_1 ;
  input alu_out_18_sp_1;
  input \alu_out[18]_0 ;
  input alu_out_15_sp_1;
  input \alu_out[15]_0 ;
  input \alu_out[15]_1 ;
  input alu_out_16_sp_1;
  input alu_out_13_sp_1;
  input \alu_out[13]_0 ;
  input \alu_out[13]_1 ;
  input alu_out_14_sp_1;
  input alu_out_11_sp_1;
  input \alu_out[11]_0 ;
  input \alu_out[11]_1 ;
  input alu_out_12_sp_1;
  input alu_out_4_sp_1;
  input \alu_out[4]_0 ;
  input \alu_out[4]_1 ;
  input alu_out_5_sp_1;
  input \alu_out[5]_0 ;
  input alu_out_6_sp_1;
  input \alu_out[6]_0 ;
  input alu_out_7_sp_1;
  input \alu_out[7]_0 ;
  input alu_out_8_sp_1;
  input \alu_out[8]_0 ;
  input alu_out_9_sp_1;
  input \alu_out[9]_0 ;
  input alu_out_10_sp_1;
  input alu_out_19_sp_1;
  input \alu_out[19]_0 ;
  input alu_out_20_sp_1;
  input \alu_out[20]_0 ;
  input alu_out_21_sp_1;
  input \alu_out[21]_0 ;
  input alu_out_22_sp_1;
  input \alu_out[22]_0 ;
  input alu_out_23_sp_1;
  input \alu_out[23]_0 ;
  input alu_out_24_sp_1;
  input \alu_out[24]_0 ;
  input alu_out_25_sp_1;
  input \alu_out[25]_0 ;
  input alu_out_26_sp_1;
  input \alu_out[26]_0 ;
  input alu_out_27_sp_1;
  input \alu_out[27]_0 ;
  input alu_out_28_sp_1;
  input \alu_out[28]_0 ;
  input alu_out_29_sp_1;
  input \alu_out[29]_0 ;
  input alu_out_30_sp_1;
  input alu_out_0_sp_1;
  input alu_out_3_sp_1;

  wire [31:0]alu_a;
  wire [31:0]alu_b;
  wire [31:0]alu_out;
  wire alu_out0__93_carry__0_n_0;
  wire alu_out0__93_carry__0_n_1;
  wire alu_out0__93_carry__0_n_2;
  wire alu_out0__93_carry__0_n_3;
  wire alu_out0__93_carry__1_n_0;
  wire alu_out0__93_carry__1_n_1;
  wire alu_out0__93_carry__1_n_2;
  wire alu_out0__93_carry__1_n_3;
  wire alu_out0__93_carry__2_n_1;
  wire alu_out0__93_carry__2_n_2;
  wire alu_out0__93_carry__2_n_3;
  wire alu_out0__93_carry_i_1__0_n_0;
  wire alu_out0__93_carry_i_1__1_n_0;
  wire alu_out0__93_carry_i_1__2_n_0;
  wire alu_out0__93_carry_i_1_n_0;
  wire alu_out0__93_carry_i_2__0_n_0;
  wire alu_out0__93_carry_i_2__1_n_0;
  wire alu_out0__93_carry_i_2__2_n_0;
  wire alu_out0__93_carry_i_2_n_0;
  wire alu_out0__93_carry_i_3__0_n_0;
  wire alu_out0__93_carry_i_3__1_n_0;
  wire alu_out0__93_carry_i_3__2_n_0;
  wire alu_out0__93_carry_i_3_n_0;
  wire alu_out0__93_carry_i_4__0_n_0;
  wire alu_out0__93_carry_i_4__1_n_0;
  wire alu_out0__93_carry_i_4__2_n_0;
  wire alu_out0__93_carry_i_4_n_0;
  wire alu_out0__93_carry_i_5__0_n_0;
  wire alu_out0__93_carry_i_5__1_n_0;
  wire alu_out0__93_carry_i_5__2_n_0;
  wire alu_out0__93_carry_i_5_n_0;
  wire alu_out0__93_carry_i_6__0_n_0;
  wire alu_out0__93_carry_i_6__1_n_0;
  wire alu_out0__93_carry_i_6__2_n_0;
  wire alu_out0__93_carry_i_6_n_0;
  wire alu_out0__93_carry_i_7__0_n_0;
  wire alu_out0__93_carry_i_7__1_n_0;
  wire alu_out0__93_carry_i_7__2_n_0;
  wire alu_out0__93_carry_i_7_n_0;
  wire alu_out0__93_carry_i_8__0_n_0;
  wire alu_out0__93_carry_i_8__1_n_0;
  wire alu_out0__93_carry_i_8__2_n_0;
  wire alu_out0__93_carry_i_8_n_0;
  wire alu_out0__93_carry_n_0;
  wire alu_out0__93_carry_n_1;
  wire alu_out0__93_carry_n_2;
  wire alu_out0__93_carry_n_3;
  wire alu_out0_carry__0_i_1_n_0;
  wire alu_out0_carry__0_i_2_n_0;
  wire alu_out0_carry__0_i_3_n_0;
  wire alu_out0_carry__0_i_4_n_0;
  wire alu_out0_carry__0_n_0;
  wire alu_out0_carry__0_n_1;
  wire alu_out0_carry__0_n_2;
  wire alu_out0_carry__0_n_3;
  wire alu_out0_carry__1_i_1_n_0;
  wire alu_out0_carry__1_i_2_n_0;
  wire alu_out0_carry__1_i_3_n_0;
  wire alu_out0_carry__1_i_4_n_0;
  wire alu_out0_carry__1_n_0;
  wire alu_out0_carry__1_n_1;
  wire alu_out0_carry__1_n_2;
  wire alu_out0_carry__1_n_3;
  wire alu_out0_carry__2_i_1_n_0;
  wire alu_out0_carry__2_i_2_n_0;
  wire alu_out0_carry__2_i_3_n_0;
  wire alu_out0_carry__2_i_4_n_0;
  wire alu_out0_carry__2_n_0;
  wire alu_out0_carry__2_n_1;
  wire alu_out0_carry__2_n_2;
  wire alu_out0_carry__2_n_3;
  wire alu_out0_carry__3_i_1_n_0;
  wire alu_out0_carry__3_i_2_n_0;
  wire alu_out0_carry__3_i_3_n_0;
  wire alu_out0_carry__3_i_4_n_0;
  wire alu_out0_carry__3_n_0;
  wire alu_out0_carry__3_n_1;
  wire alu_out0_carry__3_n_2;
  wire alu_out0_carry__3_n_3;
  wire alu_out0_carry__4_i_1_n_0;
  wire alu_out0_carry__4_i_2_n_0;
  wire alu_out0_carry__4_i_3_n_0;
  wire alu_out0_carry__4_i_4_n_0;
  wire alu_out0_carry__4_n_0;
  wire alu_out0_carry__4_n_1;
  wire alu_out0_carry__4_n_2;
  wire alu_out0_carry__4_n_3;
  wire alu_out0_carry__5_i_1_n_0;
  wire alu_out0_carry__5_i_2_n_0;
  wire alu_out0_carry__5_i_3_n_0;
  wire alu_out0_carry__5_i_4_n_0;
  wire alu_out0_carry__5_n_0;
  wire alu_out0_carry__5_n_1;
  wire alu_out0_carry__5_n_2;
  wire alu_out0_carry__5_n_3;
  wire alu_out0_carry__6_i_1_n_0;
  wire alu_out0_carry__6_i_2_n_0;
  wire alu_out0_carry__6_i_3_n_0;
  wire alu_out0_carry__6_i_4_n_0;
  wire alu_out0_carry__6_n_1;
  wire alu_out0_carry__6_n_2;
  wire alu_out0_carry__6_n_3;
  wire alu_out0_carry_i_1_n_0;
  wire alu_out0_carry_i_2_n_0;
  wire alu_out0_carry_i_3_n_0;
  wire alu_out0_carry_i_4_n_0;
  wire alu_out0_carry_n_0;
  wire alu_out0_carry_n_1;
  wire alu_out0_carry_n_2;
  wire alu_out0_carry_n_3;
  wire \alu_out[0]_INST_0_i_2_n_0 ;
  wire \alu_out[0]_INST_0_i_3_n_0 ;
  wire \alu_out[10]_INST_0_i_3_n_0 ;
  wire \alu_out[11]_0 ;
  wire \alu_out[11]_1 ;
  wire \alu_out[11]_INST_0_i_3_n_0 ;
  wire \alu_out[12]_INST_0_i_3_n_0 ;
  wire \alu_out[13]_0 ;
  wire \alu_out[13]_1 ;
  wire \alu_out[13]_INST_0_i_3_n_0 ;
  wire \alu_out[14]_INST_0_i_3_n_0 ;
  wire \alu_out[15]_0 ;
  wire \alu_out[15]_1 ;
  wire \alu_out[15]_INST_0_i_3_n_0 ;
  wire \alu_out[16]_INST_0_i_3_n_0 ;
  wire \alu_out[17]_0 ;
  wire \alu_out[17]_1 ;
  wire \alu_out[17]_INST_0_i_3_n_0 ;
  wire \alu_out[18]_0 ;
  wire \alu_out[18]_INST_0_i_3_n_0 ;
  wire \alu_out[19]_0 ;
  wire \alu_out[19]_INST_0_i_3_n_0 ;
  wire \alu_out[1]_0 ;
  wire \alu_out[1]_1 ;
  wire \alu_out[1]_INST_0_i_3_n_0 ;
  wire \alu_out[20]_0 ;
  wire \alu_out[20]_INST_0_i_3_n_0 ;
  wire \alu_out[21]_0 ;
  wire \alu_out[21]_INST_0_i_3_n_0 ;
  wire \alu_out[22]_0 ;
  wire \alu_out[22]_INST_0_i_3_n_0 ;
  wire \alu_out[23]_0 ;
  wire \alu_out[23]_INST_0_i_3_n_0 ;
  wire \alu_out[24]_0 ;
  wire \alu_out[24]_INST_0_i_3_n_0 ;
  wire \alu_out[25]_0 ;
  wire \alu_out[25]_INST_0_i_3_n_0 ;
  wire \alu_out[26]_0 ;
  wire \alu_out[26]_INST_0_i_3_n_0 ;
  wire \alu_out[27]_0 ;
  wire \alu_out[27]_INST_0_i_3_n_0 ;
  wire \alu_out[28]_0 ;
  wire \alu_out[28]_INST_0_i_3_n_0 ;
  wire \alu_out[29]_0 ;
  wire \alu_out[29]_INST_0_i_3_n_0 ;
  wire \alu_out[2]_0 ;
  wire \alu_out[2]_INST_0_i_3_n_0 ;
  wire \alu_out[30]_INST_0_i_3_n_0 ;
  wire \alu_out[31]_0 ;
  wire \alu_out[31]_1 ;
  wire \alu_out[31]_2 ;
  wire \alu_out[31]_INST_0_i_5_n_0 ;
  wire \alu_out[3]_INST_0_i_3_n_0 ;
  wire \alu_out[4]_0 ;
  wire \alu_out[4]_1 ;
  wire \alu_out[4]_INST_0_i_3_n_0 ;
  wire \alu_out[5]_0 ;
  wire \alu_out[5]_INST_0_i_3_n_0 ;
  wire \alu_out[6]_0 ;
  wire \alu_out[6]_INST_0_i_3_n_0 ;
  wire \alu_out[7]_0 ;
  wire \alu_out[7]_INST_0_i_3_n_0 ;
  wire \alu_out[8]_0 ;
  wire \alu_out[8]_INST_0_i_3_n_0 ;
  wire \alu_out[9]_0 ;
  wire \alu_out[9]_INST_0_i_3_n_0 ;
  wire alu_out_0_sn_1;
  wire alu_out_10_sn_1;
  wire alu_out_11_sn_1;
  wire alu_out_12_sn_1;
  wire alu_out_13_sn_1;
  wire alu_out_14_sn_1;
  wire alu_out_15_sn_1;
  wire alu_out_16_sn_1;
  wire alu_out_17_sn_1;
  wire alu_out_18_sn_1;
  wire alu_out_19_sn_1;
  wire alu_out_1_sn_1;
  wire alu_out_20_sn_1;
  wire alu_out_21_sn_1;
  wire alu_out_22_sn_1;
  wire alu_out_23_sn_1;
  wire alu_out_24_sn_1;
  wire alu_out_25_sn_1;
  wire alu_out_26_sn_1;
  wire alu_out_27_sn_1;
  wire alu_out_28_sn_1;
  wire alu_out_29_sn_1;
  wire alu_out_2_sn_1;
  wire alu_out_30_sn_1;
  wire alu_out_31_sn_1;
  wire alu_out_3_sn_1;
  wire alu_out_4_sn_1;
  wire alu_out_5_sn_1;
  wire alu_out_6_sn_1;
  wire alu_out_7_sn_1;
  wire alu_out_8_sn_1;
  wire alu_out_9_sn_1;
  wire [2:0]alu_sel;
  wire [31:0]data0;
  wire data4;
  wire [3:0]NLW_alu_out0__93_carry_O_UNCONNECTED;
  wire [3:0]NLW_alu_out0__93_carry__0_O_UNCONNECTED;
  wire [3:0]NLW_alu_out0__93_carry__1_O_UNCONNECTED;
  wire [3:0]NLW_alu_out0__93_carry__2_O_UNCONNECTED;
  wire [3:3]NLW_alu_out0_carry__6_CO_UNCONNECTED;

  assign alu_out_0_sn_1 = alu_out_0_sp_1;
  assign alu_out_10_sn_1 = alu_out_10_sp_1;
  assign alu_out_11_sn_1 = alu_out_11_sp_1;
  assign alu_out_12_sn_1 = alu_out_12_sp_1;
  assign alu_out_13_sn_1 = alu_out_13_sp_1;
  assign alu_out_14_sn_1 = alu_out_14_sp_1;
  assign alu_out_15_sn_1 = alu_out_15_sp_1;
  assign alu_out_16_sn_1 = alu_out_16_sp_1;
  assign alu_out_17_sn_1 = alu_out_17_sp_1;
  assign alu_out_18_sn_1 = alu_out_18_sp_1;
  assign alu_out_19_sn_1 = alu_out_19_sp_1;
  assign alu_out_1_sn_1 = alu_out_1_sp_1;
  assign alu_out_20_sn_1 = alu_out_20_sp_1;
  assign alu_out_21_sn_1 = alu_out_21_sp_1;
  assign alu_out_22_sn_1 = alu_out_22_sp_1;
  assign alu_out_23_sn_1 = alu_out_23_sp_1;
  assign alu_out_24_sn_1 = alu_out_24_sp_1;
  assign alu_out_25_sn_1 = alu_out_25_sp_1;
  assign alu_out_26_sn_1 = alu_out_26_sp_1;
  assign alu_out_27_sn_1 = alu_out_27_sp_1;
  assign alu_out_28_sn_1 = alu_out_28_sp_1;
  assign alu_out_29_sn_1 = alu_out_29_sp_1;
  assign alu_out_2_sn_1 = alu_out_2_sp_1;
  assign alu_out_30_sn_1 = alu_out_30_sp_1;
  assign alu_out_31_sn_1 = alu_out_31_sp_1;
  assign alu_out_3_sn_1 = alu_out_3_sp_1;
  assign alu_out_4_sn_1 = alu_out_4_sp_1;
  assign alu_out_5_sn_1 = alu_out_5_sp_1;
  assign alu_out_6_sn_1 = alu_out_6_sp_1;
  assign alu_out_7_sn_1 = alu_out_7_sp_1;
  assign alu_out_8_sn_1 = alu_out_8_sp_1;
  assign alu_out_9_sn_1 = alu_out_9_sp_1;
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 alu_out0__93_carry
       (.CI(1'b0),
        .CO({alu_out0__93_carry_n_0,alu_out0__93_carry_n_1,alu_out0__93_carry_n_2,alu_out0__93_carry_n_3}),
        .CYINIT(1'b0),
        .DI({alu_out0__93_carry_i_1_n_0,alu_out0__93_carry_i_2_n_0,alu_out0__93_carry_i_3_n_0,alu_out0__93_carry_i_4_n_0}),
        .O(NLW_alu_out0__93_carry_O_UNCONNECTED[3:0]),
        .S({alu_out0__93_carry_i_5_n_0,alu_out0__93_carry_i_6_n_0,alu_out0__93_carry_i_7_n_0,alu_out0__93_carry_i_8_n_0}));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 alu_out0__93_carry__0
       (.CI(alu_out0__93_carry_n_0),
        .CO({alu_out0__93_carry__0_n_0,alu_out0__93_carry__0_n_1,alu_out0__93_carry__0_n_2,alu_out0__93_carry__0_n_3}),
        .CYINIT(1'b0),
        .DI({alu_out0__93_carry_i_1__0_n_0,alu_out0__93_carry_i_2__0_n_0,alu_out0__93_carry_i_3__0_n_0,alu_out0__93_carry_i_4__0_n_0}),
        .O(NLW_alu_out0__93_carry__0_O_UNCONNECTED[3:0]),
        .S({alu_out0__93_carry_i_5__0_n_0,alu_out0__93_carry_i_6__0_n_0,alu_out0__93_carry_i_7__0_n_0,alu_out0__93_carry_i_8__0_n_0}));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 alu_out0__93_carry__1
       (.CI(alu_out0__93_carry__0_n_0),
        .CO({alu_out0__93_carry__1_n_0,alu_out0__93_carry__1_n_1,alu_out0__93_carry__1_n_2,alu_out0__93_carry__1_n_3}),
        .CYINIT(1'b0),
        .DI({alu_out0__93_carry_i_1__1_n_0,alu_out0__93_carry_i_2__1_n_0,alu_out0__93_carry_i_3__1_n_0,alu_out0__93_carry_i_4__1_n_0}),
        .O(NLW_alu_out0__93_carry__1_O_UNCONNECTED[3:0]),
        .S({alu_out0__93_carry_i_5__1_n_0,alu_out0__93_carry_i_6__1_n_0,alu_out0__93_carry_i_7__1_n_0,alu_out0__93_carry_i_8__1_n_0}));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 alu_out0__93_carry__2
       (.CI(alu_out0__93_carry__1_n_0),
        .CO({data4,alu_out0__93_carry__2_n_1,alu_out0__93_carry__2_n_2,alu_out0__93_carry__2_n_3}),
        .CYINIT(1'b0),
        .DI({alu_out0__93_carry_i_1__2_n_0,alu_out0__93_carry_i_2__2_n_0,alu_out0__93_carry_i_3__2_n_0,alu_out0__93_carry_i_4__2_n_0}),
        .O(NLW_alu_out0__93_carry__2_O_UNCONNECTED[3:0]),
        .S({alu_out0__93_carry_i_5__2_n_0,alu_out0__93_carry_i_6__2_n_0,alu_out0__93_carry_i_7__2_n_0,alu_out0__93_carry_i_8__2_n_0}));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_1
       (.I0(alu_a[6]),
        .I1(alu_b[6]),
        .I2(alu_a[7]),
        .I3(alu_b[7]),
        .O(alu_out0__93_carry_i_1_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_1__0
       (.I0(alu_a[14]),
        .I1(alu_b[14]),
        .I2(alu_a[15]),
        .I3(alu_b[15]),
        .O(alu_out0__93_carry_i_1__0_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_1__1
       (.I0(alu_a[22]),
        .I1(alu_b[22]),
        .I2(alu_a[23]),
        .I3(alu_b[23]),
        .O(alu_out0__93_carry_i_1__1_n_0));
  LUT4 #(
    .INIT(16'h44D4)) 
    alu_out0__93_carry_i_1__2
       (.I0(alu_b[31]),
        .I1(alu_a[31]),
        .I2(alu_b[30]),
        .I3(alu_a[30]),
        .O(alu_out0__93_carry_i_1__2_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_2
       (.I0(alu_a[4]),
        .I1(alu_b[4]),
        .I2(alu_a[5]),
        .I3(alu_b[5]),
        .O(alu_out0__93_carry_i_2_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_2__0
       (.I0(alu_a[12]),
        .I1(alu_b[12]),
        .I2(alu_a[13]),
        .I3(alu_b[13]),
        .O(alu_out0__93_carry_i_2__0_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_2__1
       (.I0(alu_a[20]),
        .I1(alu_b[20]),
        .I2(alu_a[21]),
        .I3(alu_b[21]),
        .O(alu_out0__93_carry_i_2__1_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_2__2
       (.I0(alu_a[28]),
        .I1(alu_b[28]),
        .I2(alu_a[29]),
        .I3(alu_b[29]),
        .O(alu_out0__93_carry_i_2__2_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_3
       (.I0(alu_a[2]),
        .I1(alu_b[2]),
        .I2(alu_a[3]),
        .I3(alu_b[3]),
        .O(alu_out0__93_carry_i_3_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_3__0
       (.I0(alu_a[10]),
        .I1(alu_b[10]),
        .I2(alu_a[11]),
        .I3(alu_b[11]),
        .O(alu_out0__93_carry_i_3__0_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_3__1
       (.I0(alu_a[18]),
        .I1(alu_b[18]),
        .I2(alu_a[19]),
        .I3(alu_b[19]),
        .O(alu_out0__93_carry_i_3__1_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_3__2
       (.I0(alu_a[26]),
        .I1(alu_b[26]),
        .I2(alu_a[27]),
        .I3(alu_b[27]),
        .O(alu_out0__93_carry_i_3__2_n_0));
  LUT4 #(
    .INIT(16'h44D4)) 
    alu_out0__93_carry_i_4
       (.I0(alu_a[1]),
        .I1(alu_b[1]),
        .I2(alu_b[0]),
        .I3(alu_a[0]),
        .O(alu_out0__93_carry_i_4_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_4__0
       (.I0(alu_a[8]),
        .I1(alu_b[8]),
        .I2(alu_a[9]),
        .I3(alu_b[9]),
        .O(alu_out0__93_carry_i_4__0_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_4__1
       (.I0(alu_a[16]),
        .I1(alu_b[16]),
        .I2(alu_a[17]),
        .I3(alu_b[17]),
        .O(alu_out0__93_carry_i_4__1_n_0));
  LUT4 #(
    .INIT(16'h4F04)) 
    alu_out0__93_carry_i_4__2
       (.I0(alu_a[24]),
        .I1(alu_b[24]),
        .I2(alu_a[25]),
        .I3(alu_b[25]),
        .O(alu_out0__93_carry_i_4__2_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_5
       (.I0(alu_b[7]),
        .I1(alu_a[7]),
        .I2(alu_b[6]),
        .I3(alu_a[6]),
        .O(alu_out0__93_carry_i_5_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_5__0
       (.I0(alu_b[15]),
        .I1(alu_a[15]),
        .I2(alu_b[14]),
        .I3(alu_a[14]),
        .O(alu_out0__93_carry_i_5__0_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_5__1
       (.I0(alu_b[23]),
        .I1(alu_a[23]),
        .I2(alu_b[22]),
        .I3(alu_a[22]),
        .O(alu_out0__93_carry_i_5__1_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_5__2
       (.I0(alu_b[30]),
        .I1(alu_a[30]),
        .I2(alu_a[31]),
        .I3(alu_b[31]),
        .O(alu_out0__93_carry_i_5__2_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_6
       (.I0(alu_b[5]),
        .I1(alu_a[5]),
        .I2(alu_b[4]),
        .I3(alu_a[4]),
        .O(alu_out0__93_carry_i_6_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_6__0
       (.I0(alu_b[13]),
        .I1(alu_a[13]),
        .I2(alu_b[12]),
        .I3(alu_a[12]),
        .O(alu_out0__93_carry_i_6__0_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_6__1
       (.I0(alu_b[21]),
        .I1(alu_a[21]),
        .I2(alu_b[20]),
        .I3(alu_a[20]),
        .O(alu_out0__93_carry_i_6__1_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_6__2
       (.I0(alu_b[29]),
        .I1(alu_a[29]),
        .I2(alu_b[28]),
        .I3(alu_a[28]),
        .O(alu_out0__93_carry_i_6__2_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_7
       (.I0(alu_b[3]),
        .I1(alu_a[3]),
        .I2(alu_b[2]),
        .I3(alu_a[2]),
        .O(alu_out0__93_carry_i_7_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_7__0
       (.I0(alu_b[11]),
        .I1(alu_a[11]),
        .I2(alu_b[10]),
        .I3(alu_a[10]),
        .O(alu_out0__93_carry_i_7__0_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_7__1
       (.I0(alu_b[19]),
        .I1(alu_a[19]),
        .I2(alu_b[18]),
        .I3(alu_a[18]),
        .O(alu_out0__93_carry_i_7__1_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_7__2
       (.I0(alu_b[27]),
        .I1(alu_a[27]),
        .I2(alu_b[26]),
        .I3(alu_a[26]),
        .O(alu_out0__93_carry_i_7__2_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_8
       (.I0(alu_b[1]),
        .I1(alu_a[1]),
        .I2(alu_b[0]),
        .I3(alu_a[0]),
        .O(alu_out0__93_carry_i_8_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_8__0
       (.I0(alu_b[9]),
        .I1(alu_a[9]),
        .I2(alu_b[8]),
        .I3(alu_a[8]),
        .O(alu_out0__93_carry_i_8__0_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_8__1
       (.I0(alu_b[17]),
        .I1(alu_a[17]),
        .I2(alu_b[16]),
        .I3(alu_a[16]),
        .O(alu_out0__93_carry_i_8__1_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    alu_out0__93_carry_i_8__2
       (.I0(alu_b[25]),
        .I1(alu_a[25]),
        .I2(alu_b[24]),
        .I3(alu_a[24]),
        .O(alu_out0__93_carry_i_8__2_n_0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 alu_out0_carry
       (.CI(1'b0),
        .CO({alu_out0_carry_n_0,alu_out0_carry_n_1,alu_out0_carry_n_2,alu_out0_carry_n_3}),
        .CYINIT(1'b0),
        .DI(alu_a[3:0]),
        .O(data0[3:0]),
        .S({alu_out0_carry_i_1_n_0,alu_out0_carry_i_2_n_0,alu_out0_carry_i_3_n_0,alu_out0_carry_i_4_n_0}));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 alu_out0_carry__0
       (.CI(alu_out0_carry_n_0),
        .CO({alu_out0_carry__0_n_0,alu_out0_carry__0_n_1,alu_out0_carry__0_n_2,alu_out0_carry__0_n_3}),
        .CYINIT(1'b0),
        .DI(alu_a[7:4]),
        .O(data0[7:4]),
        .S({alu_out0_carry__0_i_1_n_0,alu_out0_carry__0_i_2_n_0,alu_out0_carry__0_i_3_n_0,alu_out0_carry__0_i_4_n_0}));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__0_i_1
       (.I0(alu_a[7]),
        .I1(alu_b[7]),
        .O(alu_out0_carry__0_i_1_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__0_i_2
       (.I0(alu_a[6]),
        .I1(alu_b[6]),
        .O(alu_out0_carry__0_i_2_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__0_i_3
       (.I0(alu_a[5]),
        .I1(alu_b[5]),
        .O(alu_out0_carry__0_i_3_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__0_i_4
       (.I0(alu_a[4]),
        .I1(alu_b[4]),
        .O(alu_out0_carry__0_i_4_n_0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 alu_out0_carry__1
       (.CI(alu_out0_carry__0_n_0),
        .CO({alu_out0_carry__1_n_0,alu_out0_carry__1_n_1,alu_out0_carry__1_n_2,alu_out0_carry__1_n_3}),
        .CYINIT(1'b0),
        .DI(alu_a[11:8]),
        .O(data0[11:8]),
        .S({alu_out0_carry__1_i_1_n_0,alu_out0_carry__1_i_2_n_0,alu_out0_carry__1_i_3_n_0,alu_out0_carry__1_i_4_n_0}));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__1_i_1
       (.I0(alu_a[11]),
        .I1(alu_b[11]),
        .O(alu_out0_carry__1_i_1_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__1_i_2
       (.I0(alu_a[10]),
        .I1(alu_b[10]),
        .O(alu_out0_carry__1_i_2_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__1_i_3
       (.I0(alu_a[9]),
        .I1(alu_b[9]),
        .O(alu_out0_carry__1_i_3_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__1_i_4
       (.I0(alu_a[8]),
        .I1(alu_b[8]),
        .O(alu_out0_carry__1_i_4_n_0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 alu_out0_carry__2
       (.CI(alu_out0_carry__1_n_0),
        .CO({alu_out0_carry__2_n_0,alu_out0_carry__2_n_1,alu_out0_carry__2_n_2,alu_out0_carry__2_n_3}),
        .CYINIT(1'b0),
        .DI(alu_a[15:12]),
        .O(data0[15:12]),
        .S({alu_out0_carry__2_i_1_n_0,alu_out0_carry__2_i_2_n_0,alu_out0_carry__2_i_3_n_0,alu_out0_carry__2_i_4_n_0}));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__2_i_1
       (.I0(alu_a[15]),
        .I1(alu_b[15]),
        .O(alu_out0_carry__2_i_1_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__2_i_2
       (.I0(alu_a[14]),
        .I1(alu_b[14]),
        .O(alu_out0_carry__2_i_2_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__2_i_3
       (.I0(alu_a[13]),
        .I1(alu_b[13]),
        .O(alu_out0_carry__2_i_3_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__2_i_4
       (.I0(alu_a[12]),
        .I1(alu_b[12]),
        .O(alu_out0_carry__2_i_4_n_0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 alu_out0_carry__3
       (.CI(alu_out0_carry__2_n_0),
        .CO({alu_out0_carry__3_n_0,alu_out0_carry__3_n_1,alu_out0_carry__3_n_2,alu_out0_carry__3_n_3}),
        .CYINIT(1'b0),
        .DI(alu_a[19:16]),
        .O(data0[19:16]),
        .S({alu_out0_carry__3_i_1_n_0,alu_out0_carry__3_i_2_n_0,alu_out0_carry__3_i_3_n_0,alu_out0_carry__3_i_4_n_0}));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__3_i_1
       (.I0(alu_a[19]),
        .I1(alu_b[19]),
        .O(alu_out0_carry__3_i_1_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__3_i_2
       (.I0(alu_a[18]),
        .I1(alu_b[18]),
        .O(alu_out0_carry__3_i_2_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__3_i_3
       (.I0(alu_a[17]),
        .I1(alu_b[17]),
        .O(alu_out0_carry__3_i_3_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__3_i_4
       (.I0(alu_a[16]),
        .I1(alu_b[16]),
        .O(alu_out0_carry__3_i_4_n_0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 alu_out0_carry__4
       (.CI(alu_out0_carry__3_n_0),
        .CO({alu_out0_carry__4_n_0,alu_out0_carry__4_n_1,alu_out0_carry__4_n_2,alu_out0_carry__4_n_3}),
        .CYINIT(1'b0),
        .DI(alu_a[23:20]),
        .O(data0[23:20]),
        .S({alu_out0_carry__4_i_1_n_0,alu_out0_carry__4_i_2_n_0,alu_out0_carry__4_i_3_n_0,alu_out0_carry__4_i_4_n_0}));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__4_i_1
       (.I0(alu_a[23]),
        .I1(alu_b[23]),
        .O(alu_out0_carry__4_i_1_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__4_i_2
       (.I0(alu_a[22]),
        .I1(alu_b[22]),
        .O(alu_out0_carry__4_i_2_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__4_i_3
       (.I0(alu_a[21]),
        .I1(alu_b[21]),
        .O(alu_out0_carry__4_i_3_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__4_i_4
       (.I0(alu_a[20]),
        .I1(alu_b[20]),
        .O(alu_out0_carry__4_i_4_n_0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 alu_out0_carry__5
       (.CI(alu_out0_carry__4_n_0),
        .CO({alu_out0_carry__5_n_0,alu_out0_carry__5_n_1,alu_out0_carry__5_n_2,alu_out0_carry__5_n_3}),
        .CYINIT(1'b0),
        .DI(alu_a[27:24]),
        .O(data0[27:24]),
        .S({alu_out0_carry__5_i_1_n_0,alu_out0_carry__5_i_2_n_0,alu_out0_carry__5_i_3_n_0,alu_out0_carry__5_i_4_n_0}));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__5_i_1
       (.I0(alu_a[27]),
        .I1(alu_b[27]),
        .O(alu_out0_carry__5_i_1_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__5_i_2
       (.I0(alu_a[26]),
        .I1(alu_b[26]),
        .O(alu_out0_carry__5_i_2_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__5_i_3
       (.I0(alu_a[25]),
        .I1(alu_b[25]),
        .O(alu_out0_carry__5_i_3_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__5_i_4
       (.I0(alu_a[24]),
        .I1(alu_b[24]),
        .O(alu_out0_carry__5_i_4_n_0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 alu_out0_carry__6
       (.CI(alu_out0_carry__5_n_0),
        .CO({NLW_alu_out0_carry__6_CO_UNCONNECTED[3],alu_out0_carry__6_n_1,alu_out0_carry__6_n_2,alu_out0_carry__6_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,alu_a[30:28]}),
        .O(data0[31:28]),
        .S({alu_out0_carry__6_i_1_n_0,alu_out0_carry__6_i_2_n_0,alu_out0_carry__6_i_3_n_0,alu_out0_carry__6_i_4_n_0}));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__6_i_1
       (.I0(alu_b[31]),
        .I1(alu_a[31]),
        .O(alu_out0_carry__6_i_1_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__6_i_2
       (.I0(alu_a[30]),
        .I1(alu_b[30]),
        .O(alu_out0_carry__6_i_2_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__6_i_3
       (.I0(alu_a[29]),
        .I1(alu_b[29]),
        .O(alu_out0_carry__6_i_3_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry__6_i_4
       (.I0(alu_a[28]),
        .I1(alu_b[28]),
        .O(alu_out0_carry__6_i_4_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry_i_1
       (.I0(alu_a[3]),
        .I1(alu_b[3]),
        .O(alu_out0_carry_i_1_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry_i_2
       (.I0(alu_a[2]),
        .I1(alu_b[2]),
        .O(alu_out0_carry_i_2_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry_i_3
       (.I0(alu_a[1]),
        .I1(alu_b[1]),
        .O(alu_out0_carry_i_3_n_0));
  LUT2 #(
    .INIT(4'h6)) 
    alu_out0_carry_i_4
       (.I0(alu_a[0]),
        .I1(alu_b[0]),
        .O(alu_out0_carry_i_4_n_0));
  LUT5 #(
    .INIT(32'hFFFF4540)) 
    \alu_out[0]_INST_0 
       (.I0(alu_sel[2]),
        .I1(alu_out_0_sn_1),
        .I2(alu_sel[1]),
        .I3(\alu_out[0]_INST_0_i_2_n_0 ),
        .I4(\alu_out[0]_INST_0_i_3_n_0 ),
        .O(alu_out[0]));
  LUT4 #(
    .INIT(16'h8F80)) 
    \alu_out[0]_INST_0_i_2 
       (.I0(alu_b[0]),
        .I1(alu_a[0]),
        .I2(alu_sel[0]),
        .I3(data0[0]),
        .O(\alu_out[0]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h006F000000600000)) 
    \alu_out[0]_INST_0_i_3 
       (.I0(alu_a[0]),
        .I1(alu_b[0]),
        .I2(alu_sel[0]),
        .I3(alu_sel[1]),
        .I4(alu_sel[2]),
        .I5(data4),
        .O(\alu_out[0]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[10]_INST_0 
       (.I0(alu_out_10_sn_1),
        .I1(\alu_out[9]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[11]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[10]_INST_0_i_3_n_0 ),
        .O(alu_out[10]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[10]_INST_0_i_3 
       (.I0(alu_a[10]),
        .I1(alu_b[10]),
        .I2(data0[10]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[10]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[11]_INST_0 
       (.I0(alu_out_11_sn_1),
        .I1(\alu_out[11]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[11]_1 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[11]_INST_0_i_3_n_0 ),
        .O(alu_out[11]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[11]_INST_0_i_3 
       (.I0(alu_a[11]),
        .I1(alu_b[11]),
        .I2(data0[11]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[11]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[12]_INST_0 
       (.I0(alu_out_12_sn_1),
        .I1(\alu_out[11]_1 ),
        .I2(alu_b[0]),
        .I3(\alu_out[13]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[12]_INST_0_i_3_n_0 ),
        .O(alu_out[12]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[12]_INST_0_i_3 
       (.I0(alu_a[12]),
        .I1(alu_b[12]),
        .I2(data0[12]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[12]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[13]_INST_0 
       (.I0(alu_out_13_sn_1),
        .I1(\alu_out[13]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[13]_1 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[13]_INST_0_i_3_n_0 ),
        .O(alu_out[13]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[13]_INST_0_i_3 
       (.I0(alu_a[13]),
        .I1(alu_b[13]),
        .I2(data0[13]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[13]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[14]_INST_0 
       (.I0(alu_out_14_sn_1),
        .I1(\alu_out[13]_1 ),
        .I2(alu_b[0]),
        .I3(\alu_out[15]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[14]_INST_0_i_3_n_0 ),
        .O(alu_out[14]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[14]_INST_0_i_3 
       (.I0(alu_a[14]),
        .I1(alu_b[14]),
        .I2(data0[14]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[14]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[15]_INST_0 
       (.I0(alu_out_15_sn_1),
        .I1(\alu_out[15]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[15]_1 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[15]_INST_0_i_3_n_0 ),
        .O(alu_out[15]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[15]_INST_0_i_3 
       (.I0(alu_a[15]),
        .I1(alu_b[15]),
        .I2(data0[15]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[15]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[16]_INST_0 
       (.I0(alu_out_16_sn_1),
        .I1(\alu_out[15]_1 ),
        .I2(alu_b[0]),
        .I3(\alu_out[17]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[16]_INST_0_i_3_n_0 ),
        .O(alu_out[16]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[16]_INST_0_i_3 
       (.I0(alu_a[16]),
        .I1(alu_b[16]),
        .I2(data0[16]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[16]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[17]_INST_0 
       (.I0(alu_out_17_sn_1),
        .I1(\alu_out[17]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[17]_1 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[17]_INST_0_i_3_n_0 ),
        .O(alu_out[17]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[17]_INST_0_i_3 
       (.I0(alu_a[17]),
        .I1(alu_b[17]),
        .I2(data0[17]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[17]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[18]_INST_0 
       (.I0(alu_out_18_sn_1),
        .I1(\alu_out[17]_1 ),
        .I2(alu_b[0]),
        .I3(\alu_out[18]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[18]_INST_0_i_3_n_0 ),
        .O(alu_out[18]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[18]_INST_0_i_3 
       (.I0(alu_a[18]),
        .I1(alu_b[18]),
        .I2(data0[18]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[18]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[19]_INST_0 
       (.I0(alu_out_19_sn_1),
        .I1(\alu_out[18]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[19]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[19]_INST_0_i_3_n_0 ),
        .O(alu_out[19]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[19]_INST_0_i_3 
       (.I0(alu_a[19]),
        .I1(alu_b[19]),
        .I2(data0[19]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[19]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[1]_INST_0 
       (.I0(alu_out_1_sn_1),
        .I1(\alu_out[1]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[1]_1 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[1]_INST_0_i_3_n_0 ),
        .O(alu_out[1]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[1]_INST_0_i_3 
       (.I0(alu_b[1]),
        .I1(alu_a[1]),
        .I2(data0[1]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[1]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[20]_INST_0 
       (.I0(alu_out_20_sn_1),
        .I1(\alu_out[19]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[20]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[20]_INST_0_i_3_n_0 ),
        .O(alu_out[20]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[20]_INST_0_i_3 
       (.I0(alu_a[20]),
        .I1(alu_b[20]),
        .I2(data0[20]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[20]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[21]_INST_0 
       (.I0(alu_out_21_sn_1),
        .I1(\alu_out[20]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[21]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[21]_INST_0_i_3_n_0 ),
        .O(alu_out[21]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[21]_INST_0_i_3 
       (.I0(alu_a[21]),
        .I1(alu_b[21]),
        .I2(data0[21]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[21]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[22]_INST_0 
       (.I0(alu_out_22_sn_1),
        .I1(\alu_out[21]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[22]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[22]_INST_0_i_3_n_0 ),
        .O(alu_out[22]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[22]_INST_0_i_3 
       (.I0(alu_a[22]),
        .I1(alu_b[22]),
        .I2(data0[22]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[22]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[23]_INST_0 
       (.I0(alu_out_23_sn_1),
        .I1(\alu_out[22]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[23]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[23]_INST_0_i_3_n_0 ),
        .O(alu_out[23]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[23]_INST_0_i_3 
       (.I0(alu_a[23]),
        .I1(alu_b[23]),
        .I2(data0[23]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[23]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[24]_INST_0 
       (.I0(alu_out_24_sn_1),
        .I1(\alu_out[23]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[24]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[24]_INST_0_i_3_n_0 ),
        .O(alu_out[24]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[24]_INST_0_i_3 
       (.I0(alu_a[24]),
        .I1(alu_b[24]),
        .I2(data0[24]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[24]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[25]_INST_0 
       (.I0(alu_out_25_sn_1),
        .I1(\alu_out[24]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[25]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[25]_INST_0_i_3_n_0 ),
        .O(alu_out[25]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[25]_INST_0_i_3 
       (.I0(alu_a[25]),
        .I1(alu_b[25]),
        .I2(data0[25]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[25]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[26]_INST_0 
       (.I0(alu_out_26_sn_1),
        .I1(\alu_out[25]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[26]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[26]_INST_0_i_3_n_0 ),
        .O(alu_out[26]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[26]_INST_0_i_3 
       (.I0(alu_a[26]),
        .I1(alu_b[26]),
        .I2(data0[26]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[26]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[27]_INST_0 
       (.I0(alu_out_27_sn_1),
        .I1(\alu_out[26]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[27]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[27]_INST_0_i_3_n_0 ),
        .O(alu_out[27]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[27]_INST_0_i_3 
       (.I0(alu_a[27]),
        .I1(alu_b[27]),
        .I2(data0[27]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[27]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[28]_INST_0 
       (.I0(alu_out_28_sn_1),
        .I1(\alu_out[27]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[28]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[28]_INST_0_i_3_n_0 ),
        .O(alu_out[28]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[28]_INST_0_i_3 
       (.I0(alu_a[28]),
        .I1(alu_b[28]),
        .I2(data0[28]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[28]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[29]_INST_0 
       (.I0(alu_out_29_sn_1),
        .I1(\alu_out[28]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[29]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[29]_INST_0_i_3_n_0 ),
        .O(alu_out[29]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[29]_INST_0_i_3 
       (.I0(alu_a[29]),
        .I1(alu_b[29]),
        .I2(data0[29]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[29]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[2]_INST_0 
       (.I0(alu_out_2_sn_1),
        .I1(\alu_out[1]_1 ),
        .I2(alu_b[0]),
        .I3(\alu_out[2]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[2]_INST_0_i_3_n_0 ),
        .O(alu_out[2]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[2]_INST_0_i_3 
       (.I0(alu_b[2]),
        .I1(alu_a[2]),
        .I2(data0[2]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[2]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[30]_INST_0 
       (.I0(alu_out_30_sn_1),
        .I1(\alu_out[29]_0 ),
        .I2(alu_b[0]),
        .I3(alu_a[31]),
        .I4(alu_sel[0]),
        .I5(\alu_out[30]_INST_0_i_3_n_0 ),
        .O(alu_out[30]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[30]_INST_0_i_3 
       (.I0(alu_b[30]),
        .I1(alu_a[30]),
        .I2(data0[30]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[30]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FFFBFBFB)) 
    \alu_out[31]_INST_0 
       (.I0(alu_out_31_sn_1),
        .I1(alu_sel[1]),
        .I2(\alu_out[31]_0 ),
        .I3(\alu_out[31]_1 ),
        .I4(\alu_out[31]_2 ),
        .I5(\alu_out[31]_INST_0_i_5_n_0 ),
        .O(alu_out[31]));
  LUT6 #(
    .INIT(64'hFFC3FFFF003F0055)) 
    \alu_out[31]_INST_0_i_5 
       (.I0(data0[31]),
        .I1(alu_a[31]),
        .I2(alu_b[31]),
        .I3(alu_sel[1]),
        .I4(alu_sel[0]),
        .I5(alu_sel[2]),
        .O(\alu_out[31]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[3]_INST_0 
       (.I0(alu_out_3_sn_1),
        .I1(\alu_out[2]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[4]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[3]_INST_0_i_3_n_0 ),
        .O(alu_out[3]));
  LUT6 #(
    .INIT(64'hF0C3C3F3F1F1F1F1)) 
    \alu_out[3]_INST_0_i_3 
       (.I0(data0[3]),
        .I1(alu_sel[1]),
        .I2(alu_sel[2]),
        .I3(alu_a[3]),
        .I4(alu_b[3]),
        .I5(alu_sel[0]),
        .O(\alu_out[3]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[4]_INST_0 
       (.I0(alu_out_4_sn_1),
        .I1(\alu_out[4]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[4]_1 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[4]_INST_0_i_3_n_0 ),
        .O(alu_out[4]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[4]_INST_0_i_3 
       (.I0(alu_b[4]),
        .I1(alu_a[4]),
        .I2(data0[4]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[4]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[5]_INST_0 
       (.I0(alu_out_5_sn_1),
        .I1(\alu_out[4]_1 ),
        .I2(alu_b[0]),
        .I3(\alu_out[5]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[5]_INST_0_i_3_n_0 ),
        .O(alu_out[5]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[5]_INST_0_i_3 
       (.I0(alu_a[5]),
        .I1(alu_b[5]),
        .I2(data0[5]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[5]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[6]_INST_0 
       (.I0(alu_out_6_sn_1),
        .I1(\alu_out[5]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[6]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[6]_INST_0_i_3_n_0 ),
        .O(alu_out[6]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[6]_INST_0_i_3 
       (.I0(alu_a[6]),
        .I1(alu_b[6]),
        .I2(data0[6]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[6]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[7]_INST_0 
       (.I0(alu_out_7_sn_1),
        .I1(\alu_out[6]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[7]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[7]_INST_0_i_3_n_0 ),
        .O(alu_out[7]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[7]_INST_0_i_3 
       (.I0(alu_a[7]),
        .I1(alu_b[7]),
        .I2(data0[7]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[7]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[8]_INST_0 
       (.I0(alu_out_8_sn_1),
        .I1(\alu_out[7]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[8]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[8]_INST_0_i_3_n_0 ),
        .O(alu_out[8]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[8]_INST_0_i_3 
       (.I0(alu_a[8]),
        .I1(alu_b[8]),
        .I2(data0[8]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[8]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FEAEAAAA)) 
    \alu_out[9]_INST_0 
       (.I0(alu_out_9_sn_1),
        .I1(\alu_out[8]_0 ),
        .I2(alu_b[0]),
        .I3(\alu_out[9]_0 ),
        .I4(alu_sel[0]),
        .I5(\alu_out[9]_INST_0_i_3_n_0 ),
        .O(alu_out[9]));
  LUT6 #(
    .INIT(64'hFFFF99FF0000770F)) 
    \alu_out[9]_INST_0_i_3 
       (.I0(alu_a[9]),
        .I1(alu_b[9]),
        .I2(data0[9]),
        .I3(alu_sel[0]),
        .I4(alu_sel[1]),
        .I5(alu_sel[2]),
        .O(\alu_out[9]_INST_0_i_3_n_0 ));
endmodule

(* CHECK_LICENSE_TYPE = "risc32_alu32_0_0,alu32,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "alu32,Vivado 2022.2" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (alu_a,
    alu_b,
    alu_sel,
    alu_out);
  input [31:0]alu_a;
  input [31:0]alu_b;
  input [2:0]alu_sel;
  output [31:0]alu_out;

  wire [31:0]alu_a;
  wire [31:0]alu_b;
  wire [31:0]alu_out;
  wire \alu_out[0]_INST_0_i_1_n_0 ;
  wire \alu_out[0]_INST_0_i_4_n_0 ;
  wire \alu_out[0]_INST_0_i_5_n_0 ;
  wire \alu_out[10]_INST_0_i_1_n_0 ;
  wire \alu_out[10]_INST_0_i_2_n_0 ;
  wire \alu_out[10]_INST_0_i_4_n_0 ;
  wire \alu_out[10]_INST_0_i_5_n_0 ;
  wire \alu_out[10]_INST_0_i_6_n_0 ;
  wire \alu_out[11]_INST_0_i_1_n_0 ;
  wire \alu_out[11]_INST_0_i_2_n_0 ;
  wire \alu_out[11]_INST_0_i_4_n_0 ;
  wire \alu_out[11]_INST_0_i_5_n_0 ;
  wire \alu_out[11]_INST_0_i_6_n_0 ;
  wire \alu_out[12]_INST_0_i_1_n_0 ;
  wire \alu_out[12]_INST_0_i_2_n_0 ;
  wire \alu_out[12]_INST_0_i_4_n_0 ;
  wire \alu_out[12]_INST_0_i_5_n_0 ;
  wire \alu_out[12]_INST_0_i_6_n_0 ;
  wire \alu_out[13]_INST_0_i_1_n_0 ;
  wire \alu_out[13]_INST_0_i_2_n_0 ;
  wire \alu_out[13]_INST_0_i_4_n_0 ;
  wire \alu_out[13]_INST_0_i_5_n_0 ;
  wire \alu_out[13]_INST_0_i_6_n_0 ;
  wire \alu_out[14]_INST_0_i_1_n_0 ;
  wire \alu_out[14]_INST_0_i_2_n_0 ;
  wire \alu_out[14]_INST_0_i_4_n_0 ;
  wire \alu_out[14]_INST_0_i_5_n_0 ;
  wire \alu_out[14]_INST_0_i_6_n_0 ;
  wire \alu_out[15]_INST_0_i_1_n_0 ;
  wire \alu_out[15]_INST_0_i_2_n_0 ;
  wire \alu_out[15]_INST_0_i_4_n_0 ;
  wire \alu_out[15]_INST_0_i_5_n_0 ;
  wire \alu_out[15]_INST_0_i_6_n_0 ;
  wire \alu_out[15]_INST_0_i_7_n_0 ;
  wire \alu_out[16]_INST_0_i_1_n_0 ;
  wire \alu_out[16]_INST_0_i_2_n_0 ;
  wire \alu_out[16]_INST_0_i_4_n_0 ;
  wire \alu_out[16]_INST_0_i_5_n_0 ;
  wire \alu_out[16]_INST_0_i_6_n_0 ;
  wire \alu_out[16]_INST_0_i_7_n_0 ;
  wire \alu_out[17]_INST_0_i_1_n_0 ;
  wire \alu_out[17]_INST_0_i_2_n_0 ;
  wire \alu_out[17]_INST_0_i_4_n_0 ;
  wire \alu_out[17]_INST_0_i_5_n_0 ;
  wire \alu_out[17]_INST_0_i_6_n_0 ;
  wire \alu_out[17]_INST_0_i_7_n_0 ;
  wire \alu_out[18]_INST_0_i_1_n_0 ;
  wire \alu_out[18]_INST_0_i_2_n_0 ;
  wire \alu_out[18]_INST_0_i_4_n_0 ;
  wire \alu_out[18]_INST_0_i_5_n_0 ;
  wire \alu_out[18]_INST_0_i_6_n_0 ;
  wire \alu_out[18]_INST_0_i_7_n_0 ;
  wire \alu_out[19]_INST_0_i_1_n_0 ;
  wire \alu_out[19]_INST_0_i_2_n_0 ;
  wire \alu_out[19]_INST_0_i_4_n_0 ;
  wire \alu_out[19]_INST_0_i_5_n_0 ;
  wire \alu_out[19]_INST_0_i_6_n_0 ;
  wire \alu_out[1]_INST_0_i_1_n_0 ;
  wire \alu_out[1]_INST_0_i_2_n_0 ;
  wire \alu_out[1]_INST_0_i_4_n_0 ;
  wire \alu_out[1]_INST_0_i_5_n_0 ;
  wire \alu_out[20]_INST_0_i_1_n_0 ;
  wire \alu_out[20]_INST_0_i_2_n_0 ;
  wire \alu_out[20]_INST_0_i_4_n_0 ;
  wire \alu_out[20]_INST_0_i_5_n_0 ;
  wire \alu_out[20]_INST_0_i_6_n_0 ;
  wire \alu_out[21]_INST_0_i_1_n_0 ;
  wire \alu_out[21]_INST_0_i_2_n_0 ;
  wire \alu_out[21]_INST_0_i_4_n_0 ;
  wire \alu_out[21]_INST_0_i_5_n_0 ;
  wire \alu_out[21]_INST_0_i_6_n_0 ;
  wire \alu_out[22]_INST_0_i_1_n_0 ;
  wire \alu_out[22]_INST_0_i_2_n_0 ;
  wire \alu_out[22]_INST_0_i_4_n_0 ;
  wire \alu_out[22]_INST_0_i_5_n_0 ;
  wire \alu_out[22]_INST_0_i_6_n_0 ;
  wire \alu_out[23]_INST_0_i_1_n_0 ;
  wire \alu_out[23]_INST_0_i_2_n_0 ;
  wire \alu_out[23]_INST_0_i_4_n_0 ;
  wire \alu_out[23]_INST_0_i_5_n_0 ;
  wire \alu_out[23]_INST_0_i_6_n_0 ;
  wire \alu_out[24]_INST_0_i_1_n_0 ;
  wire \alu_out[24]_INST_0_i_2_n_0 ;
  wire \alu_out[24]_INST_0_i_4_n_0 ;
  wire \alu_out[24]_INST_0_i_5_n_0 ;
  wire \alu_out[24]_INST_0_i_6_n_0 ;
  wire \alu_out[25]_INST_0_i_1_n_0 ;
  wire \alu_out[25]_INST_0_i_2_n_0 ;
  wire \alu_out[25]_INST_0_i_4_n_0 ;
  wire \alu_out[25]_INST_0_i_5_n_0 ;
  wire \alu_out[25]_INST_0_i_6_n_0 ;
  wire \alu_out[26]_INST_0_i_1_n_0 ;
  wire \alu_out[26]_INST_0_i_2_n_0 ;
  wire \alu_out[26]_INST_0_i_4_n_0 ;
  wire \alu_out[26]_INST_0_i_5_n_0 ;
  wire \alu_out[26]_INST_0_i_6_n_0 ;
  wire \alu_out[27]_INST_0_i_1_n_0 ;
  wire \alu_out[27]_INST_0_i_2_n_0 ;
  wire \alu_out[27]_INST_0_i_4_n_0 ;
  wire \alu_out[27]_INST_0_i_5_n_0 ;
  wire \alu_out[27]_INST_0_i_6_n_0 ;
  wire \alu_out[27]_INST_0_i_7_n_0 ;
  wire \alu_out[28]_INST_0_i_1_n_0 ;
  wire \alu_out[28]_INST_0_i_2_n_0 ;
  wire \alu_out[28]_INST_0_i_4_n_0 ;
  wire \alu_out[28]_INST_0_i_5_n_0 ;
  wire \alu_out[28]_INST_0_i_6_n_0 ;
  wire \alu_out[28]_INST_0_i_7_n_0 ;
  wire \alu_out[29]_INST_0_i_1_n_0 ;
  wire \alu_out[29]_INST_0_i_2_n_0 ;
  wire \alu_out[29]_INST_0_i_4_n_0 ;
  wire \alu_out[29]_INST_0_i_5_n_0 ;
  wire \alu_out[2]_INST_0_i_1_n_0 ;
  wire \alu_out[2]_INST_0_i_2_n_0 ;
  wire \alu_out[2]_INST_0_i_4_n_0 ;
  wire \alu_out[2]_INST_0_i_5_n_0 ;
  wire \alu_out[30]_INST_0_i_1_n_0 ;
  wire \alu_out[30]_INST_0_i_2_n_0 ;
  wire \alu_out[30]_INST_0_i_4_n_0 ;
  wire \alu_out[30]_INST_0_i_5_n_0 ;
  wire \alu_out[31]_INST_0_i_10_n_0 ;
  wire \alu_out[31]_INST_0_i_11_n_0 ;
  wire \alu_out[31]_INST_0_i_12_n_0 ;
  wire \alu_out[31]_INST_0_i_13_n_0 ;
  wire \alu_out[31]_INST_0_i_14_n_0 ;
  wire \alu_out[31]_INST_0_i_15_n_0 ;
  wire \alu_out[31]_INST_0_i_1_n_0 ;
  wire \alu_out[31]_INST_0_i_2_n_0 ;
  wire \alu_out[31]_INST_0_i_3_n_0 ;
  wire \alu_out[31]_INST_0_i_4_n_0 ;
  wire \alu_out[31]_INST_0_i_6_n_0 ;
  wire \alu_out[31]_INST_0_i_7_n_0 ;
  wire \alu_out[31]_INST_0_i_8_n_0 ;
  wire \alu_out[31]_INST_0_i_9_n_0 ;
  wire \alu_out[3]_INST_0_i_1_n_0 ;
  wire \alu_out[3]_INST_0_i_2_n_0 ;
  wire \alu_out[3]_INST_0_i_4_n_0 ;
  wire \alu_out[3]_INST_0_i_5_n_0 ;
  wire \alu_out[4]_INST_0_i_1_n_0 ;
  wire \alu_out[4]_INST_0_i_2_n_0 ;
  wire \alu_out[4]_INST_0_i_4_n_0 ;
  wire \alu_out[4]_INST_0_i_5_n_0 ;
  wire \alu_out[5]_INST_0_i_1_n_0 ;
  wire \alu_out[5]_INST_0_i_2_n_0 ;
  wire \alu_out[5]_INST_0_i_4_n_0 ;
  wire \alu_out[5]_INST_0_i_5_n_0 ;
  wire \alu_out[6]_INST_0_i_1_n_0 ;
  wire \alu_out[6]_INST_0_i_2_n_0 ;
  wire \alu_out[6]_INST_0_i_4_n_0 ;
  wire \alu_out[6]_INST_0_i_5_n_0 ;
  wire \alu_out[7]_INST_0_i_1_n_0 ;
  wire \alu_out[7]_INST_0_i_2_n_0 ;
  wire \alu_out[7]_INST_0_i_4_n_0 ;
  wire \alu_out[7]_INST_0_i_5_n_0 ;
  wire \alu_out[7]_INST_0_i_6_n_0 ;
  wire \alu_out[8]_INST_0_i_1_n_0 ;
  wire \alu_out[8]_INST_0_i_2_n_0 ;
  wire \alu_out[8]_INST_0_i_4_n_0 ;
  wire \alu_out[8]_INST_0_i_5_n_0 ;
  wire \alu_out[8]_INST_0_i_6_n_0 ;
  wire \alu_out[9]_INST_0_i_1_n_0 ;
  wire \alu_out[9]_INST_0_i_2_n_0 ;
  wire \alu_out[9]_INST_0_i_4_n_0 ;
  wire \alu_out[9]_INST_0_i_5_n_0 ;
  wire \alu_out[9]_INST_0_i_6_n_0 ;
  wire [2:0]alu_sel;

  LUT5 #(
    .INIT(32'hA0C0A0CF)) 
    \alu_out[0]_INST_0_i_1 
       (.I0(\alu_out[1]_INST_0_i_2_n_0 ),
        .I1(\alu_out[0]_INST_0_i_4_n_0 ),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[1]_INST_0_i_4_n_0 ),
        .O(\alu_out[0]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[0]_INST_0_i_4 
       (.I0(\alu_out[6]_INST_0_i_5_n_0 ),
        .I1(\alu_out[2]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[4]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[0]_INST_0_i_5_n_0 ),
        .O(\alu_out[0]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[0]_INST_0_i_5 
       (.I0(alu_a[24]),
        .I1(alu_a[8]),
        .I2(alu_b[3]),
        .I3(alu_a[16]),
        .I4(alu_b[4]),
        .I5(alu_a[0]),
        .O(\alu_out[0]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[10]_INST_0_i_1 
       (.I0(\alu_out[11]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[10]_INST_0_i_4_n_0 ),
        .O(\alu_out[10]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[10]_INST_0_i_2 
       (.I0(\alu_out[16]_INST_0_i_6_n_0 ),
        .I1(\alu_out[12]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[14]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[10]_INST_0_i_5_n_0 ),
        .O(\alu_out[10]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[10]_INST_0_i_4 
       (.I0(\alu_out[10]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[12]_INST_0_i_6_n_0 ),
        .O(\alu_out[10]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[10]_INST_0_i_5 
       (.I0(alu_a[31]),
        .I1(alu_a[18]),
        .I2(alu_b[3]),
        .I3(alu_a[26]),
        .I4(alu_b[4]),
        .I5(alu_a[10]),
        .O(\alu_out[10]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hFFF4FFF7)) 
    \alu_out[10]_INST_0_i_6 
       (.I0(alu_a[3]),
        .I1(alu_b[2]),
        .I2(alu_b[3]),
        .I3(alu_b[4]),
        .I4(alu_a[7]),
        .O(\alu_out[10]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[11]_INST_0_i_1 
       (.I0(\alu_out[12]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[11]_INST_0_i_4_n_0 ),
        .O(\alu_out[11]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[11]_INST_0_i_2 
       (.I0(\alu_out[17]_INST_0_i_6_n_0 ),
        .I1(\alu_out[13]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[15]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[11]_INST_0_i_5_n_0 ),
        .O(\alu_out[11]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[11]_INST_0_i_4 
       (.I0(\alu_out[11]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[13]_INST_0_i_6_n_0 ),
        .O(\alu_out[11]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[11]_INST_0_i_5 
       (.I0(alu_a[31]),
        .I1(alu_a[19]),
        .I2(alu_b[3]),
        .I3(alu_a[27]),
        .I4(alu_b[4]),
        .I5(alu_a[11]),
        .O(\alu_out[11]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFCF44CF77)) 
    \alu_out[11]_INST_0_i_6 
       (.I0(alu_a[4]),
        .I1(alu_b[2]),
        .I2(alu_a[0]),
        .I3(alu_b[3]),
        .I4(alu_a[8]),
        .I5(alu_b[4]),
        .O(\alu_out[11]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[12]_INST_0_i_1 
       (.I0(\alu_out[13]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[12]_INST_0_i_4_n_0 ),
        .O(\alu_out[12]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[12]_INST_0_i_2 
       (.I0(\alu_out[18]_INST_0_i_6_n_0 ),
        .I1(\alu_out[14]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[16]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[12]_INST_0_i_5_n_0 ),
        .O(\alu_out[12]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[12]_INST_0_i_4 
       (.I0(\alu_out[12]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[14]_INST_0_i_6_n_0 ),
        .O(\alu_out[12]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[12]_INST_0_i_5 
       (.I0(alu_a[31]),
        .I1(alu_a[20]),
        .I2(alu_b[3]),
        .I3(alu_a[28]),
        .I4(alu_b[4]),
        .I5(alu_a[12]),
        .O(\alu_out[12]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFCF44CF77)) 
    \alu_out[12]_INST_0_i_6 
       (.I0(alu_a[5]),
        .I1(alu_b[2]),
        .I2(alu_a[1]),
        .I3(alu_b[3]),
        .I4(alu_a[9]),
        .I5(alu_b[4]),
        .O(\alu_out[12]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[13]_INST_0_i_1 
       (.I0(\alu_out[14]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[13]_INST_0_i_4_n_0 ),
        .O(\alu_out[13]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[13]_INST_0_i_2 
       (.I0(\alu_out[15]_INST_0_i_5_n_0 ),
        .I1(\alu_out[15]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[17]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[13]_INST_0_i_5_n_0 ),
        .O(\alu_out[13]_INST_0_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \alu_out[13]_INST_0_i_4 
       (.I0(\alu_out[13]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[15]_INST_0_i_7_n_0 ),
        .I3(alu_b[2]),
        .I4(\alu_out[19]_INST_0_i_6_n_0 ),
        .O(\alu_out[13]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[13]_INST_0_i_5 
       (.I0(alu_a[31]),
        .I1(alu_a[21]),
        .I2(alu_b[3]),
        .I3(alu_a[29]),
        .I4(alu_b[4]),
        .I5(alu_a[13]),
        .O(\alu_out[13]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFCF44CF77)) 
    \alu_out[13]_INST_0_i_6 
       (.I0(alu_a[6]),
        .I1(alu_b[2]),
        .I2(alu_a[2]),
        .I3(alu_b[3]),
        .I4(alu_a[10]),
        .I5(alu_b[4]),
        .O(\alu_out[13]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[14]_INST_0_i_1 
       (.I0(\alu_out[15]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[14]_INST_0_i_4_n_0 ),
        .O(\alu_out[14]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[14]_INST_0_i_2 
       (.I0(\alu_out[16]_INST_0_i_5_n_0 ),
        .I1(\alu_out[16]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[18]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[14]_INST_0_i_5_n_0 ),
        .O(\alu_out[14]_INST_0_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \alu_out[14]_INST_0_i_4 
       (.I0(\alu_out[14]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[16]_INST_0_i_7_n_0 ),
        .I3(alu_b[2]),
        .I4(\alu_out[20]_INST_0_i_6_n_0 ),
        .O(\alu_out[14]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[14]_INST_0_i_5 
       (.I0(alu_a[31]),
        .I1(alu_a[22]),
        .I2(alu_b[3]),
        .I3(alu_a[30]),
        .I4(alu_b[4]),
        .I5(alu_a[14]),
        .O(\alu_out[14]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFCF44CF77)) 
    \alu_out[14]_INST_0_i_6 
       (.I0(alu_a[7]),
        .I1(alu_b[2]),
        .I2(alu_a[3]),
        .I3(alu_b[3]),
        .I4(alu_a[11]),
        .I5(alu_b[4]),
        .O(\alu_out[14]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[15]_INST_0_i_1 
       (.I0(\alu_out[16]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[15]_INST_0_i_4_n_0 ),
        .O(\alu_out[15]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[15]_INST_0_i_2 
       (.I0(\alu_out[17]_INST_0_i_5_n_0 ),
        .I1(\alu_out[17]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[15]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[15]_INST_0_i_6_n_0 ),
        .O(\alu_out[15]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFAFC0CFA0A0C0CF)) 
    \alu_out[15]_INST_0_i_4 
       (.I0(\alu_out[15]_INST_0_i_7_n_0 ),
        .I1(\alu_out[19]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[21]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[17]_INST_0_i_7_n_0 ),
        .O(\alu_out[15]_INST_0_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \alu_out[15]_INST_0_i_5 
       (.I0(alu_a[27]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[19]),
        .O(\alu_out[15]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \alu_out[15]_INST_0_i_6 
       (.I0(alu_a[23]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[15]),
        .O(\alu_out[15]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT4 #(
    .INIT(16'hFF47)) 
    \alu_out[15]_INST_0_i_7 
       (.I0(alu_a[0]),
        .I1(alu_b[3]),
        .I2(alu_a[8]),
        .I3(alu_b[4]),
        .O(\alu_out[15]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'h333B3F3B)) 
    \alu_out[16]_INST_0_i_1 
       (.I0(\alu_out[17]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[16]_INST_0_i_4_n_0 ),
        .O(\alu_out[16]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[16]_INST_0_i_2 
       (.I0(\alu_out[18]_INST_0_i_5_n_0 ),
        .I1(\alu_out[18]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[16]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[16]_INST_0_i_6_n_0 ),
        .O(\alu_out[16]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hA0AFC0C0A0AFCFCF)) 
    \alu_out[16]_INST_0_i_4 
       (.I0(\alu_out[16]_INST_0_i_7_n_0 ),
        .I1(\alu_out[20]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[18]_INST_0_i_7_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[22]_INST_0_i_6_n_0 ),
        .O(\alu_out[16]_INST_0_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \alu_out[16]_INST_0_i_5 
       (.I0(alu_a[28]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[20]),
        .O(\alu_out[16]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \alu_out[16]_INST_0_i_6 
       (.I0(alu_a[24]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[16]),
        .O(\alu_out[16]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT4 #(
    .INIT(16'hFF47)) 
    \alu_out[16]_INST_0_i_7 
       (.I0(alu_a[1]),
        .I1(alu_b[3]),
        .I2(alu_a[9]),
        .I3(alu_b[4]),
        .O(\alu_out[16]_INST_0_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[17]_INST_0_i_1 
       (.I0(\alu_out[18]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[17]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[17]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF00B8B8)) 
    \alu_out[17]_INST_0_i_2 
       (.I0(\alu_out[17]_INST_0_i_5_n_0 ),
        .I1(alu_b[2]),
        .I2(\alu_out[17]_INST_0_i_6_n_0 ),
        .I3(\alu_out[19]_INST_0_i_5_n_0 ),
        .I4(alu_b[1]),
        .O(\alu_out[17]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h505FCFCF505FC0C0)) 
    \alu_out[17]_INST_0_i_4 
       (.I0(\alu_out[17]_INST_0_i_7_n_0 ),
        .I1(\alu_out[21]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[19]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[23]_INST_0_i_6_n_0 ),
        .O(\alu_out[17]_INST_0_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \alu_out[17]_INST_0_i_5 
       (.I0(alu_a[29]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[21]),
        .O(\alu_out[17]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \alu_out[17]_INST_0_i_6 
       (.I0(alu_a[25]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[17]),
        .O(\alu_out[17]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT4 #(
    .INIT(16'hFF47)) 
    \alu_out[17]_INST_0_i_7 
       (.I0(alu_a[2]),
        .I1(alu_b[3]),
        .I2(alu_a[10]),
        .I3(alu_b[4]),
        .O(\alu_out[17]_INST_0_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[18]_INST_0_i_1 
       (.I0(\alu_out[19]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[18]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[18]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF00B8B8)) 
    \alu_out[18]_INST_0_i_2 
       (.I0(\alu_out[18]_INST_0_i_5_n_0 ),
        .I1(alu_b[2]),
        .I2(\alu_out[18]_INST_0_i_6_n_0 ),
        .I3(\alu_out[20]_INST_0_i_5_n_0 ),
        .I4(alu_b[1]),
        .O(\alu_out[18]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hA0AFCFCFA0AFC0C0)) 
    \alu_out[18]_INST_0_i_4 
       (.I0(\alu_out[18]_INST_0_i_7_n_0 ),
        .I1(\alu_out[22]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[20]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[24]_INST_0_i_6_n_0 ),
        .O(\alu_out[18]_INST_0_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \alu_out[18]_INST_0_i_5 
       (.I0(alu_a[30]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[22]),
        .O(\alu_out[18]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \alu_out[18]_INST_0_i_6 
       (.I0(alu_a[26]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[18]),
        .O(\alu_out[18]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT4 #(
    .INIT(16'h00B8)) 
    \alu_out[18]_INST_0_i_7 
       (.I0(alu_a[3]),
        .I1(alu_b[3]),
        .I2(alu_a[11]),
        .I3(alu_b[4]),
        .O(\alu_out[18]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[19]_INST_0_i_1 
       (.I0(\alu_out[20]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[19]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[19]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[19]_INST_0_i_2 
       (.I0(\alu_out[21]_INST_0_i_5_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[19]_INST_0_i_5_n_0 ),
        .O(\alu_out[19]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h5F50CFCF5F50C0C0)) 
    \alu_out[19]_INST_0_i_4 
       (.I0(\alu_out[19]_INST_0_i_6_n_0 ),
        .I1(\alu_out[23]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[21]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[25]_INST_0_i_6_n_0 ),
        .O(\alu_out[19]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8FFFFCDC80000)) 
    \alu_out[19]_INST_0_i_5 
       (.I0(alu_b[3]),
        .I1(alu_a[31]),
        .I2(alu_b[4]),
        .I3(alu_a[23]),
        .I4(alu_b[2]),
        .I5(\alu_out[15]_INST_0_i_5_n_0 ),
        .O(\alu_out[19]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT4 #(
    .INIT(16'hFF47)) 
    \alu_out[19]_INST_0_i_6 
       (.I0(alu_a[4]),
        .I1(alu_b[3]),
        .I2(alu_a[12]),
        .I3(alu_b[4]),
        .O(\alu_out[19]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h55555F77)) 
    \alu_out[1]_INST_0_i_1 
       (.I0(alu_sel[1]),
        .I1(\alu_out[2]_INST_0_i_4_n_0 ),
        .I2(\alu_out[1]_INST_0_i_4_n_0 ),
        .I3(alu_b[0]),
        .I4(alu_sel[0]),
        .O(\alu_out[1]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[1]_INST_0_i_2 
       (.I0(\alu_out[7]_INST_0_i_5_n_0 ),
        .I1(\alu_out[3]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[5]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[1]_INST_0_i_5_n_0 ),
        .O(\alu_out[1]_INST_0_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFEF)) 
    \alu_out[1]_INST_0_i_4 
       (.I0(alu_b[1]),
        .I1(alu_b[3]),
        .I2(alu_a[0]),
        .I3(alu_b[4]),
        .I4(alu_b[2]),
        .O(\alu_out[1]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[1]_INST_0_i_5 
       (.I0(alu_a[25]),
        .I1(alu_a[9]),
        .I2(alu_b[3]),
        .I3(alu_a[17]),
        .I4(alu_b[4]),
        .I5(alu_a[1]),
        .O(\alu_out[1]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[20]_INST_0_i_1 
       (.I0(\alu_out[21]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[20]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[20]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[20]_INST_0_i_2 
       (.I0(\alu_out[22]_INST_0_i_5_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[20]_INST_0_i_5_n_0 ),
        .O(\alu_out[20]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h5F50CFCF5F50C0C0)) 
    \alu_out[20]_INST_0_i_4 
       (.I0(\alu_out[20]_INST_0_i_6_n_0 ),
        .I1(\alu_out[24]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[22]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[26]_INST_0_i_6_n_0 ),
        .O(\alu_out[20]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8FFFFCDC80000)) 
    \alu_out[20]_INST_0_i_5 
       (.I0(alu_b[3]),
        .I1(alu_a[31]),
        .I2(alu_b[4]),
        .I3(alu_a[24]),
        .I4(alu_b[2]),
        .I5(\alu_out[16]_INST_0_i_5_n_0 ),
        .O(\alu_out[20]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT4 #(
    .INIT(16'hFF47)) 
    \alu_out[20]_INST_0_i_6 
       (.I0(alu_a[5]),
        .I1(alu_b[3]),
        .I2(alu_a[13]),
        .I3(alu_b[4]),
        .O(\alu_out[20]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[21]_INST_0_i_1 
       (.I0(\alu_out[22]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[21]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[21]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[21]_INST_0_i_2 
       (.I0(\alu_out[23]_INST_0_i_5_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[21]_INST_0_i_5_n_0 ),
        .O(\alu_out[21]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[21]_INST_0_i_4 
       (.I0(\alu_out[21]_INST_0_i_6_n_0 ),
        .I1(\alu_out[25]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[23]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[27]_INST_0_i_7_n_0 ),
        .O(\alu_out[21]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8FFFFCDC80000)) 
    \alu_out[21]_INST_0_i_5 
       (.I0(alu_b[3]),
        .I1(alu_a[31]),
        .I2(alu_b[4]),
        .I3(alu_a[25]),
        .I4(alu_b[2]),
        .I5(\alu_out[17]_INST_0_i_5_n_0 ),
        .O(\alu_out[21]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT4 #(
    .INIT(16'h00B8)) 
    \alu_out[21]_INST_0_i_6 
       (.I0(alu_a[6]),
        .I1(alu_b[3]),
        .I2(alu_a[14]),
        .I3(alu_b[4]),
        .O(\alu_out[21]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[22]_INST_0_i_1 
       (.I0(\alu_out[23]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[22]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[22]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[22]_INST_0_i_2 
       (.I0(\alu_out[24]_INST_0_i_5_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[22]_INST_0_i_5_n_0 ),
        .O(\alu_out[22]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[22]_INST_0_i_4 
       (.I0(\alu_out[22]_INST_0_i_6_n_0 ),
        .I1(\alu_out[26]_INST_0_i_6_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[24]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[28]_INST_0_i_7_n_0 ),
        .O(\alu_out[22]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8FFFFCDC80000)) 
    \alu_out[22]_INST_0_i_5 
       (.I0(alu_b[3]),
        .I1(alu_a[31]),
        .I2(alu_b[4]),
        .I3(alu_a[26]),
        .I4(alu_b[2]),
        .I5(\alu_out[18]_INST_0_i_5_n_0 ),
        .O(\alu_out[22]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT4 #(
    .INIT(16'h00B8)) 
    \alu_out[22]_INST_0_i_6 
       (.I0(alu_a[7]),
        .I1(alu_b[3]),
        .I2(alu_a[15]),
        .I3(alu_b[4]),
        .O(\alu_out[22]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[23]_INST_0_i_1 
       (.I0(\alu_out[24]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[23]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[23]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[23]_INST_0_i_2 
       (.I0(\alu_out[25]_INST_0_i_5_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[23]_INST_0_i_5_n_0 ),
        .O(\alu_out[23]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[23]_INST_0_i_4 
       (.I0(\alu_out[23]_INST_0_i_6_n_0 ),
        .I1(\alu_out[27]_INST_0_i_7_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[25]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[29]_INST_0_i_5_n_0 ),
        .O(\alu_out[23]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFF00FB0BFF00F808)) 
    \alu_out[23]_INST_0_i_5 
       (.I0(alu_a[27]),
        .I1(alu_b[2]),
        .I2(alu_b[3]),
        .I3(alu_a[31]),
        .I4(alu_b[4]),
        .I5(alu_a[23]),
        .O(\alu_out[23]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \alu_out[23]_INST_0_i_6 
       (.I0(alu_a[8]),
        .I1(alu_b[3]),
        .I2(alu_a[0]),
        .I3(alu_b[4]),
        .I4(alu_a[16]),
        .O(\alu_out[23]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[24]_INST_0_i_1 
       (.I0(\alu_out[25]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[24]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[24]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[24]_INST_0_i_2 
       (.I0(\alu_out[26]_INST_0_i_5_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[24]_INST_0_i_5_n_0 ),
        .O(\alu_out[24]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[24]_INST_0_i_4 
       (.I0(\alu_out[24]_INST_0_i_6_n_0 ),
        .I1(\alu_out[28]_INST_0_i_7_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[26]_INST_0_i_6_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[30]_INST_0_i_5_n_0 ),
        .O(\alu_out[24]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFF00FB0BFF00F808)) 
    \alu_out[24]_INST_0_i_5 
       (.I0(alu_a[28]),
        .I1(alu_b[2]),
        .I2(alu_b[3]),
        .I3(alu_a[31]),
        .I4(alu_b[4]),
        .I5(alu_a[24]),
        .O(\alu_out[24]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \alu_out[24]_INST_0_i_6 
       (.I0(alu_a[9]),
        .I1(alu_b[3]),
        .I2(alu_a[1]),
        .I3(alu_b[4]),
        .I4(alu_a[17]),
        .O(\alu_out[24]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[25]_INST_0_i_1 
       (.I0(\alu_out[26]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[25]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[25]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[25]_INST_0_i_2 
       (.I0(\alu_out[27]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[25]_INST_0_i_5_n_0 ),
        .O(\alu_out[25]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[25]_INST_0_i_4 
       (.I0(\alu_out[25]_INST_0_i_6_n_0 ),
        .I1(\alu_out[29]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[27]_INST_0_i_7_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[31]_INST_0_i_10_n_0 ),
        .O(\alu_out[25]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFF00FB0BFF00F808)) 
    \alu_out[25]_INST_0_i_5 
       (.I0(alu_a[29]),
        .I1(alu_b[2]),
        .I2(alu_b[3]),
        .I3(alu_a[31]),
        .I4(alu_b[4]),
        .I5(alu_a[25]),
        .O(\alu_out[25]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \alu_out[25]_INST_0_i_6 
       (.I0(alu_a[10]),
        .I1(alu_b[3]),
        .I2(alu_a[2]),
        .I3(alu_b[4]),
        .I4(alu_a[18]),
        .O(\alu_out[25]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[26]_INST_0_i_1 
       (.I0(\alu_out[27]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[26]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[26]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[26]_INST_0_i_2 
       (.I0(\alu_out[28]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[26]_INST_0_i_5_n_0 ),
        .O(\alu_out[26]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[26]_INST_0_i_4 
       (.I0(\alu_out[26]_INST_0_i_6_n_0 ),
        .I1(\alu_out[30]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[28]_INST_0_i_7_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[31]_INST_0_i_14_n_0 ),
        .O(\alu_out[26]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFF00FB0BFF00F808)) 
    \alu_out[26]_INST_0_i_5 
       (.I0(alu_a[30]),
        .I1(alu_b[2]),
        .I2(alu_b[3]),
        .I3(alu_a[31]),
        .I4(alu_b[4]),
        .I5(alu_a[26]),
        .O(\alu_out[26]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \alu_out[26]_INST_0_i_6 
       (.I0(alu_a[11]),
        .I1(alu_b[3]),
        .I2(alu_a[3]),
        .I3(alu_b[4]),
        .I4(alu_a[19]),
        .O(\alu_out[26]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[27]_INST_0_i_1 
       (.I0(\alu_out[28]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[27]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[27]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[27]_INST_0_i_2 
       (.I0(\alu_out[27]_INST_0_i_5_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[27]_INST_0_i_6_n_0 ),
        .O(\alu_out[27]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[27]_INST_0_i_4 
       (.I0(\alu_out[27]_INST_0_i_7_n_0 ),
        .I1(\alu_out[31]_INST_0_i_10_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[29]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[31]_INST_0_i_12_n_0 ),
        .O(\alu_out[27]_INST_0_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hF0F1F0E0)) 
    \alu_out[27]_INST_0_i_5 
       (.I0(alu_b[2]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[29]),
        .O(\alu_out[27]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hF0F1F0E0)) 
    \alu_out[27]_INST_0_i_6 
       (.I0(alu_b[2]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[27]),
        .O(\alu_out[27]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT5 #(
    .INIT(32'h3300B8B8)) 
    \alu_out[27]_INST_0_i_7 
       (.I0(alu_a[12]),
        .I1(alu_b[3]),
        .I2(alu_a[20]),
        .I3(alu_a[4]),
        .I4(alu_b[4]),
        .O(\alu_out[27]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[28]_INST_0_i_1 
       (.I0(\alu_out[29]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[28]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[28]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[28]_INST_0_i_2 
       (.I0(\alu_out[28]_INST_0_i_5_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[28]_INST_0_i_6_n_0 ),
        .O(\alu_out[28]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[28]_INST_0_i_4 
       (.I0(\alu_out[28]_INST_0_i_7_n_0 ),
        .I1(\alu_out[31]_INST_0_i_14_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[30]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[31]_INST_0_i_8_n_0 ),
        .O(\alu_out[28]_INST_0_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hF0F1F0E0)) 
    \alu_out[28]_INST_0_i_5 
       (.I0(alu_b[2]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[30]),
        .O(\alu_out[28]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hF0F1F0E0)) 
    \alu_out[28]_INST_0_i_6 
       (.I0(alu_b[2]),
        .I1(alu_b[3]),
        .I2(alu_a[31]),
        .I3(alu_b[4]),
        .I4(alu_a[28]),
        .O(\alu_out[28]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \alu_out[28]_INST_0_i_7 
       (.I0(alu_a[13]),
        .I1(alu_b[3]),
        .I2(alu_a[5]),
        .I3(alu_b[4]),
        .I4(alu_a[21]),
        .O(\alu_out[28]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[29]_INST_0_i_1 
       (.I0(\alu_out[30]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[29]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[29]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFF00FF01FF00FE00)) 
    \alu_out[29]_INST_0_i_2 
       (.I0(alu_b[1]),
        .I1(alu_b[2]),
        .I2(alu_b[3]),
        .I3(alu_a[31]),
        .I4(alu_b[4]),
        .I5(alu_a[29]),
        .O(\alu_out[29]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[29]_INST_0_i_4 
       (.I0(\alu_out[29]_INST_0_i_5_n_0 ),
        .I1(\alu_out[31]_INST_0_i_12_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[31]_INST_0_i_10_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[31]_INST_0_i_11_n_0 ),
        .O(\alu_out[29]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \alu_out[29]_INST_0_i_5 
       (.I0(alu_a[14]),
        .I1(alu_b[3]),
        .I2(alu_a[6]),
        .I3(alu_b[4]),
        .I4(alu_a[22]),
        .O(\alu_out[29]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[2]_INST_0_i_1 
       (.I0(\alu_out[3]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[2]_INST_0_i_4_n_0 ),
        .O(\alu_out[2]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFF33CC00B8B8B8B8)) 
    \alu_out[2]_INST_0_i_2 
       (.I0(\alu_out[6]_INST_0_i_5_n_0 ),
        .I1(alu_b[2]),
        .I2(\alu_out[2]_INST_0_i_5_n_0 ),
        .I3(\alu_out[8]_INST_0_i_5_n_0 ),
        .I4(\alu_out[4]_INST_0_i_5_n_0 ),
        .I5(alu_b[1]),
        .O(\alu_out[2]_INST_0_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFEF)) 
    \alu_out[2]_INST_0_i_4 
       (.I0(alu_b[1]),
        .I1(alu_b[3]),
        .I2(alu_a[1]),
        .I3(alu_b[4]),
        .I4(alu_b[2]),
        .O(\alu_out[2]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[2]_INST_0_i_5 
       (.I0(alu_a[26]),
        .I1(alu_a[10]),
        .I2(alu_b[3]),
        .I3(alu_a[18]),
        .I4(alu_b[4]),
        .I5(alu_a[2]),
        .O(\alu_out[2]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h33F333BB)) 
    \alu_out[30]_INST_0_i_1 
       (.I0(\alu_out[31]_INST_0_i_3_n_0 ),
        .I1(alu_sel[1]),
        .I2(\alu_out[30]_INST_0_i_4_n_0 ),
        .I3(alu_sel[0]),
        .I4(alu_b[0]),
        .O(\alu_out[30]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFF00FF01FF00FE00)) 
    \alu_out[30]_INST_0_i_2 
       (.I0(alu_b[1]),
        .I1(alu_b[2]),
        .I2(alu_b[3]),
        .I3(alu_a[31]),
        .I4(alu_b[4]),
        .I5(alu_a[30]),
        .O(\alu_out[30]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[30]_INST_0_i_4 
       (.I0(\alu_out[30]_INST_0_i_5_n_0 ),
        .I1(\alu_out[31]_INST_0_i_8_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[31]_INST_0_i_14_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[31]_INST_0_i_15_n_0 ),
        .O(\alu_out[30]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \alu_out[30]_INST_0_i_5 
       (.I0(alu_a[15]),
        .I1(alu_b[3]),
        .I2(alu_a[7]),
        .I3(alu_b[4]),
        .I4(alu_a[23]),
        .O(\alu_out[30]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hAAA888A822200020)) 
    \alu_out[31]_INST_0_i_1 
       (.I0(\alu_out[31]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[31]_INST_0_i_7_n_0 ),
        .I3(alu_b[2]),
        .I4(\alu_out[31]_INST_0_i_8_n_0 ),
        .I5(\alu_out[31]_INST_0_i_9_n_0 ),
        .O(\alu_out[31]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[31]_INST_0_i_10 
       (.I0(alu_a[0]),
        .I1(alu_a[16]),
        .I2(alu_b[3]),
        .I3(alu_a[8]),
        .I4(alu_b[4]),
        .I5(alu_a[24]),
        .O(\alu_out[31]_INST_0_i_10_n_0 ));
  LUT6 #(
    .INIT(64'hCFC0AFAFCFC0A0A0)) 
    \alu_out[31]_INST_0_i_11 
       (.I0(alu_a[20]),
        .I1(alu_a[4]),
        .I2(alu_b[3]),
        .I3(alu_a[12]),
        .I4(alu_b[4]),
        .I5(alu_a[28]),
        .O(\alu_out[31]_INST_0_i_11_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[31]_INST_0_i_12 
       (.I0(alu_a[2]),
        .I1(alu_a[18]),
        .I2(alu_b[3]),
        .I3(alu_a[10]),
        .I4(alu_b[4]),
        .I5(alu_a[26]),
        .O(\alu_out[31]_INST_0_i_12_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[31]_INST_0_i_13 
       (.I0(alu_a[6]),
        .I1(alu_a[22]),
        .I2(alu_b[3]),
        .I3(alu_a[14]),
        .I4(alu_b[4]),
        .I5(alu_a[30]),
        .O(\alu_out[31]_INST_0_i_13_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[31]_INST_0_i_14 
       (.I0(alu_a[1]),
        .I1(alu_a[17]),
        .I2(alu_b[3]),
        .I3(alu_a[9]),
        .I4(alu_b[4]),
        .I5(alu_a[25]),
        .O(\alu_out[31]_INST_0_i_14_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[31]_INST_0_i_15 
       (.I0(alu_a[5]),
        .I1(alu_a[21]),
        .I2(alu_b[3]),
        .I3(alu_a[13]),
        .I4(alu_b[4]),
        .I5(alu_a[29]),
        .O(\alu_out[31]_INST_0_i_15_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \alu_out[31]_INST_0_i_2 
       (.I0(alu_a[31]),
        .I1(alu_sel[0]),
        .O(\alu_out[31]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[31]_INST_0_i_3 
       (.I0(\alu_out[31]_INST_0_i_10_n_0 ),
        .I1(\alu_out[31]_INST_0_i_11_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[31]_INST_0_i_12_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[31]_INST_0_i_13_n_0 ),
        .O(\alu_out[31]_INST_0_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \alu_out[31]_INST_0_i_4 
       (.I0(alu_b[0]),
        .I1(alu_sel[0]),
        .O(\alu_out[31]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT2 #(
    .INIT(4'h1)) 
    \alu_out[31]_INST_0_i_6 
       (.I0(alu_b[0]),
        .I1(alu_sel[0]),
        .O(\alu_out[31]_INST_0_i_6_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[31]_INST_0_i_7 
       (.I0(alu_a[7]),
        .I1(alu_a[23]),
        .I2(alu_b[3]),
        .I3(alu_a[15]),
        .I4(alu_b[4]),
        .I5(alu_a[31]),
        .O(\alu_out[31]_INST_0_i_7_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[31]_INST_0_i_8 
       (.I0(alu_a[3]),
        .I1(alu_a[19]),
        .I2(alu_b[3]),
        .I3(alu_a[11]),
        .I4(alu_b[4]),
        .I5(alu_a[27]),
        .O(\alu_out[31]_INST_0_i_8_n_0 ));
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[31]_INST_0_i_9 
       (.I0(\alu_out[31]_INST_0_i_14_n_0 ),
        .I1(alu_b[2]),
        .I2(\alu_out[31]_INST_0_i_15_n_0 ),
        .O(\alu_out[31]_INST_0_i_9_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[3]_INST_0_i_1 
       (.I0(\alu_out[4]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[3]_INST_0_i_4_n_0 ),
        .O(\alu_out[3]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFF33CC00B8B8B8B8)) 
    \alu_out[3]_INST_0_i_2 
       (.I0(\alu_out[7]_INST_0_i_5_n_0 ),
        .I1(alu_b[2]),
        .I2(\alu_out[3]_INST_0_i_5_n_0 ),
        .I3(\alu_out[9]_INST_0_i_5_n_0 ),
        .I4(\alu_out[5]_INST_0_i_5_n_0 ),
        .I5(alu_b[1]),
        .O(\alu_out[3]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFF4FFF7)) 
    \alu_out[3]_INST_0_i_4 
       (.I0(alu_a[0]),
        .I1(alu_b[1]),
        .I2(alu_b[2]),
        .I3(alu_b[4]),
        .I4(alu_a[2]),
        .I5(alu_b[3]),
        .O(\alu_out[3]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[3]_INST_0_i_5 
       (.I0(alu_a[27]),
        .I1(alu_a[11]),
        .I2(alu_b[3]),
        .I3(alu_a[19]),
        .I4(alu_b[4]),
        .I5(alu_a[3]),
        .O(\alu_out[3]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[4]_INST_0_i_1 
       (.I0(\alu_out[5]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[4]_INST_0_i_4_n_0 ),
        .O(\alu_out[4]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[4]_INST_0_i_2 
       (.I0(\alu_out[10]_INST_0_i_5_n_0 ),
        .I1(\alu_out[6]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[8]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[4]_INST_0_i_5_n_0 ),
        .O(\alu_out[4]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFF4FFF7)) 
    \alu_out[4]_INST_0_i_4 
       (.I0(alu_a[1]),
        .I1(alu_b[1]),
        .I2(alu_b[2]),
        .I3(alu_b[3]),
        .I4(alu_a[3]),
        .I5(alu_b[4]),
        .O(\alu_out[4]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[4]_INST_0_i_5 
       (.I0(alu_a[28]),
        .I1(alu_a[12]),
        .I2(alu_b[3]),
        .I3(alu_a[20]),
        .I4(alu_b[4]),
        .I5(alu_a[4]),
        .O(\alu_out[4]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[5]_INST_0_i_1 
       (.I0(\alu_out[6]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[5]_INST_0_i_4_n_0 ),
        .O(\alu_out[5]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[5]_INST_0_i_2 
       (.I0(\alu_out[11]_INST_0_i_5_n_0 ),
        .I1(\alu_out[7]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[9]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[5]_INST_0_i_5_n_0 ),
        .O(\alu_out[5]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFEFFFFFFFEF0000)) 
    \alu_out[5]_INST_0_i_4 
       (.I0(alu_b[2]),
        .I1(alu_b[4]),
        .I2(alu_a[2]),
        .I3(alu_b[3]),
        .I4(alu_b[1]),
        .I5(\alu_out[7]_INST_0_i_6_n_0 ),
        .O(\alu_out[5]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[5]_INST_0_i_5 
       (.I0(alu_a[29]),
        .I1(alu_a[13]),
        .I2(alu_b[3]),
        .I3(alu_a[21]),
        .I4(alu_b[4]),
        .I5(alu_a[5]),
        .O(\alu_out[5]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[6]_INST_0_i_1 
       (.I0(\alu_out[7]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[6]_INST_0_i_4_n_0 ),
        .O(\alu_out[6]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[6]_INST_0_i_2 
       (.I0(\alu_out[12]_INST_0_i_5_n_0 ),
        .I1(\alu_out[8]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[10]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[6]_INST_0_i_5_n_0 ),
        .O(\alu_out[6]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFEFFFFFFFEF0000)) 
    \alu_out[6]_INST_0_i_4 
       (.I0(alu_b[2]),
        .I1(alu_b[3]),
        .I2(alu_a[3]),
        .I3(alu_b[4]),
        .I4(alu_b[1]),
        .I5(\alu_out[8]_INST_0_i_6_n_0 ),
        .O(\alu_out[6]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[6]_INST_0_i_5 
       (.I0(alu_a[30]),
        .I1(alu_a[14]),
        .I2(alu_b[3]),
        .I3(alu_a[22]),
        .I4(alu_b[4]),
        .I5(alu_a[6]),
        .O(\alu_out[6]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[7]_INST_0_i_1 
       (.I0(\alu_out[8]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[7]_INST_0_i_4_n_0 ),
        .O(\alu_out[7]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[7]_INST_0_i_2 
       (.I0(\alu_out[13]_INST_0_i_5_n_0 ),
        .I1(\alu_out[9]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[11]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[7]_INST_0_i_5_n_0 ),
        .O(\alu_out[7]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[7]_INST_0_i_4 
       (.I0(\alu_out[7]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[9]_INST_0_i_6_n_0 ),
        .O(\alu_out[7]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[7]_INST_0_i_5 
       (.I0(alu_a[31]),
        .I1(alu_a[15]),
        .I2(alu_b[3]),
        .I3(alu_a[23]),
        .I4(alu_b[4]),
        .I5(alu_a[7]),
        .O(\alu_out[7]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFF4F7)) 
    \alu_out[7]_INST_0_i_6 
       (.I0(alu_a[0]),
        .I1(alu_b[2]),
        .I2(alu_b[3]),
        .I3(alu_a[4]),
        .I4(alu_b[4]),
        .O(\alu_out[7]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[8]_INST_0_i_1 
       (.I0(\alu_out[9]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[8]_INST_0_i_4_n_0 ),
        .O(\alu_out[8]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[8]_INST_0_i_2 
       (.I0(\alu_out[14]_INST_0_i_5_n_0 ),
        .I1(\alu_out[10]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[12]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[8]_INST_0_i_5_n_0 ),
        .O(\alu_out[8]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[8]_INST_0_i_4 
       (.I0(\alu_out[8]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[10]_INST_0_i_6_n_0 ),
        .O(\alu_out[8]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[8]_INST_0_i_5 
       (.I0(alu_a[31]),
        .I1(alu_a[16]),
        .I2(alu_b[3]),
        .I3(alu_a[24]),
        .I4(alu_b[4]),
        .I5(alu_a[8]),
        .O(\alu_out[8]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFF4F7)) 
    \alu_out[8]_INST_0_i_6 
       (.I0(alu_a[1]),
        .I1(alu_b[2]),
        .I2(alu_b[3]),
        .I3(alu_a[5]),
        .I4(alu_b[4]),
        .O(\alu_out[8]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'h33373F37)) 
    \alu_out[9]_INST_0_i_1 
       (.I0(\alu_out[10]_INST_0_i_4_n_0 ),
        .I1(alu_sel[1]),
        .I2(alu_sel[0]),
        .I3(alu_b[0]),
        .I4(\alu_out[9]_INST_0_i_4_n_0 ),
        .O(\alu_out[9]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[9]_INST_0_i_2 
       (.I0(\alu_out[15]_INST_0_i_6_n_0 ),
        .I1(\alu_out[11]_INST_0_i_5_n_0 ),
        .I2(alu_b[1]),
        .I3(\alu_out[13]_INST_0_i_5_n_0 ),
        .I4(alu_b[2]),
        .I5(\alu_out[9]_INST_0_i_5_n_0 ),
        .O(\alu_out[9]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \alu_out[9]_INST_0_i_4 
       (.I0(\alu_out[9]_INST_0_i_6_n_0 ),
        .I1(alu_b[1]),
        .I2(\alu_out[11]_INST_0_i_6_n_0 ),
        .O(\alu_out[9]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \alu_out[9]_INST_0_i_5 
       (.I0(alu_a[31]),
        .I1(alu_a[17]),
        .I2(alu_b[3]),
        .I3(alu_a[25]),
        .I4(alu_b[4]),
        .I5(alu_a[9]),
        .O(\alu_out[9]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hFFF4FFF7)) 
    \alu_out[9]_INST_0_i_6 
       (.I0(alu_a[2]),
        .I1(alu_b[2]),
        .I2(alu_b[3]),
        .I3(alu_b[4]),
        .I4(alu_a[6]),
        .O(\alu_out[9]_INST_0_i_6_n_0 ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_alu32 inst
       (.alu_a(alu_a),
        .alu_b(alu_b),
        .alu_out(alu_out),
        .\alu_out[11]_0 (\alu_out[11]_INST_0_i_2_n_0 ),
        .\alu_out[11]_1 (\alu_out[12]_INST_0_i_2_n_0 ),
        .\alu_out[13]_0 (\alu_out[13]_INST_0_i_2_n_0 ),
        .\alu_out[13]_1 (\alu_out[14]_INST_0_i_2_n_0 ),
        .\alu_out[15]_0 (\alu_out[15]_INST_0_i_2_n_0 ),
        .\alu_out[15]_1 (\alu_out[16]_INST_0_i_2_n_0 ),
        .\alu_out[17]_0 (\alu_out[17]_INST_0_i_2_n_0 ),
        .\alu_out[17]_1 (\alu_out[18]_INST_0_i_2_n_0 ),
        .\alu_out[18]_0 (\alu_out[19]_INST_0_i_2_n_0 ),
        .\alu_out[19]_0 (\alu_out[20]_INST_0_i_2_n_0 ),
        .\alu_out[1]_0 (\alu_out[1]_INST_0_i_2_n_0 ),
        .\alu_out[1]_1 (\alu_out[2]_INST_0_i_2_n_0 ),
        .\alu_out[20]_0 (\alu_out[21]_INST_0_i_2_n_0 ),
        .\alu_out[21]_0 (\alu_out[22]_INST_0_i_2_n_0 ),
        .\alu_out[22]_0 (\alu_out[23]_INST_0_i_2_n_0 ),
        .\alu_out[23]_0 (\alu_out[24]_INST_0_i_2_n_0 ),
        .\alu_out[24]_0 (\alu_out[25]_INST_0_i_2_n_0 ),
        .\alu_out[25]_0 (\alu_out[26]_INST_0_i_2_n_0 ),
        .\alu_out[26]_0 (\alu_out[27]_INST_0_i_2_n_0 ),
        .\alu_out[27]_0 (\alu_out[28]_INST_0_i_2_n_0 ),
        .\alu_out[28]_0 (\alu_out[29]_INST_0_i_2_n_0 ),
        .\alu_out[29]_0 (\alu_out[30]_INST_0_i_2_n_0 ),
        .\alu_out[2]_0 (\alu_out[3]_INST_0_i_2_n_0 ),
        .\alu_out[31]_0 (\alu_out[31]_INST_0_i_2_n_0 ),
        .\alu_out[31]_1 (\alu_out[31]_INST_0_i_3_n_0 ),
        .\alu_out[31]_2 (\alu_out[31]_INST_0_i_4_n_0 ),
        .\alu_out[4]_0 (\alu_out[4]_INST_0_i_2_n_0 ),
        .\alu_out[4]_1 (\alu_out[5]_INST_0_i_2_n_0 ),
        .\alu_out[5]_0 (\alu_out[6]_INST_0_i_2_n_0 ),
        .\alu_out[6]_0 (\alu_out[7]_INST_0_i_2_n_0 ),
        .\alu_out[7]_0 (\alu_out[8]_INST_0_i_2_n_0 ),
        .\alu_out[8]_0 (\alu_out[9]_INST_0_i_2_n_0 ),
        .\alu_out[9]_0 (\alu_out[10]_INST_0_i_2_n_0 ),
        .alu_out_0_sp_1(\alu_out[0]_INST_0_i_1_n_0 ),
        .alu_out_10_sp_1(\alu_out[10]_INST_0_i_1_n_0 ),
        .alu_out_11_sp_1(\alu_out[11]_INST_0_i_1_n_0 ),
        .alu_out_12_sp_1(\alu_out[12]_INST_0_i_1_n_0 ),
        .alu_out_13_sp_1(\alu_out[13]_INST_0_i_1_n_0 ),
        .alu_out_14_sp_1(\alu_out[14]_INST_0_i_1_n_0 ),
        .alu_out_15_sp_1(\alu_out[15]_INST_0_i_1_n_0 ),
        .alu_out_16_sp_1(\alu_out[16]_INST_0_i_1_n_0 ),
        .alu_out_17_sp_1(\alu_out[17]_INST_0_i_1_n_0 ),
        .alu_out_18_sp_1(\alu_out[18]_INST_0_i_1_n_0 ),
        .alu_out_19_sp_1(\alu_out[19]_INST_0_i_1_n_0 ),
        .alu_out_1_sp_1(\alu_out[1]_INST_0_i_1_n_0 ),
        .alu_out_20_sp_1(\alu_out[20]_INST_0_i_1_n_0 ),
        .alu_out_21_sp_1(\alu_out[21]_INST_0_i_1_n_0 ),
        .alu_out_22_sp_1(\alu_out[22]_INST_0_i_1_n_0 ),
        .alu_out_23_sp_1(\alu_out[23]_INST_0_i_1_n_0 ),
        .alu_out_24_sp_1(\alu_out[24]_INST_0_i_1_n_0 ),
        .alu_out_25_sp_1(\alu_out[25]_INST_0_i_1_n_0 ),
        .alu_out_26_sp_1(\alu_out[26]_INST_0_i_1_n_0 ),
        .alu_out_27_sp_1(\alu_out[27]_INST_0_i_1_n_0 ),
        .alu_out_28_sp_1(\alu_out[28]_INST_0_i_1_n_0 ),
        .alu_out_29_sp_1(\alu_out[29]_INST_0_i_1_n_0 ),
        .alu_out_2_sp_1(\alu_out[2]_INST_0_i_1_n_0 ),
        .alu_out_30_sp_1(\alu_out[30]_INST_0_i_1_n_0 ),
        .alu_out_31_sp_1(\alu_out[31]_INST_0_i_1_n_0 ),
        .alu_out_3_sp_1(\alu_out[3]_INST_0_i_1_n_0 ),
        .alu_out_4_sp_1(\alu_out[4]_INST_0_i_1_n_0 ),
        .alu_out_5_sp_1(\alu_out[5]_INST_0_i_1_n_0 ),
        .alu_out_6_sp_1(\alu_out[6]_INST_0_i_1_n_0 ),
        .alu_out_7_sp_1(\alu_out[7]_INST_0_i_1_n_0 ),
        .alu_out_8_sp_1(\alu_out[8]_INST_0_i_1_n_0 ),
        .alu_out_9_sp_1(\alu_out[9]_INST_0_i_1_n_0 ),
        .alu_sel(alu_sel));
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
