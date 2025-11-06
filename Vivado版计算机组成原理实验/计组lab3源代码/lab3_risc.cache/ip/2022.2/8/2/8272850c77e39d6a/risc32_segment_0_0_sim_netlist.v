// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Tue Oct 29 10:56:54 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_segment_0_0_sim_netlist.v
// Design      : risc32_segment_0_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CHECK_LICENSE_TYPE = "risc32_segment_0_0,segment,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "segment,Vivado 2022.2" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (clk,
    rst_n,
    Data_i,
    AN,
    seg_data_o);
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk, FREQ_HZ 50000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN /clk_wiz_0_clk_out1, INSERT_VIP 0" *) input clk;
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME rst_n, POLARITY ACTIVE_LOW, INSERT_VIP 0" *) input rst_n;
  input [31:0]Data_i;
  output [7:0]AN;
  output [7:0]seg_data_o;

  wire \<const1> ;
  wire [7:0]AN;
  wire [31:0]Data_i;
  wire clk;
  wire rst_n;
  wire [6:0]\^seg_data_o ;

  assign seg_data_o[7] = \<const1> ;
  assign seg_data_o[6:0] = \^seg_data_o [6:0];
  VCC VCC
       (.P(\<const1> ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_segment inst
       (.AN(AN),
        .Data_i(Data_i),
        .clk(clk),
        .rst_n(rst_n),
        .seg_data_o(\^seg_data_o ));
endmodule

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_segment
   (AN,
    seg_data_o,
    clk,
    Data_i,
    rst_n);
  output [7:0]AN;
  output [6:0]seg_data_o;
  input clk;
  input [31:0]Data_i;
  input rst_n;

  wire [7:0]AN;
  wire \AN[0]_i_1_n_0 ;
  wire \AN[1]_i_1_n_0 ;
  wire \AN[2]_i_1_n_0 ;
  wire \AN[3]_i_1_n_0 ;
  wire \AN[4]_i_1_n_0 ;
  wire \AN[5]_i_1_n_0 ;
  wire \AN[6]_i_1_n_0 ;
  wire \AN[7]_i_1_n_0 ;
  wire \AN[7]_i_2_n_0 ;
  wire [31:0]Data_i;
  wire clk;
  wire [2:0]cnt;
  wire \cnt[0]_i_1_n_0 ;
  wire \cnt[1]_i_1_n_0 ;
  wire \cnt[2]_i_1_n_0 ;
  wire [15:0]cnt_ms;
  wire cnt_ms0_carry__0_n_0;
  wire cnt_ms0_carry__0_n_1;
  wire cnt_ms0_carry__0_n_2;
  wire cnt_ms0_carry__0_n_3;
  wire cnt_ms0_carry__1_n_0;
  wire cnt_ms0_carry__1_n_1;
  wire cnt_ms0_carry__1_n_2;
  wire cnt_ms0_carry__1_n_3;
  wire cnt_ms0_carry__2_n_2;
  wire cnt_ms0_carry__2_n_3;
  wire cnt_ms0_carry_n_0;
  wire cnt_ms0_carry_n_1;
  wire cnt_ms0_carry_n_2;
  wire cnt_ms0_carry_n_3;
  wire \cnt_ms[15]_i_2_n_0 ;
  wire \cnt_ms[15]_i_3_n_0 ;
  wire \cnt_ms[15]_i_4_n_0 ;
  wire \cnt_ms[15]_i_5_n_0 ;
  wire \cnt_ms[15]_i_6_n_0 ;
  wire [15:0]cnt_ms_1;
  wire [15:1]data0;
  wire rst_n;
  wire [6:0]seg_data_o;
  wire [6:0]seg_data_o_0;
  wire [3:0]seg_num;
  wire \seg_num[0]_i_2_n_0 ;
  wire \seg_num[0]_i_3_n_0 ;
  wire \seg_num[1]_i_2_n_0 ;
  wire \seg_num[1]_i_3_n_0 ;
  wire \seg_num[2]_i_2_n_0 ;
  wire \seg_num[2]_i_3_n_0 ;
  wire \seg_num[3]_i_2_n_0 ;
  wire \seg_num[3]_i_3_n_0 ;
  wire [3:0]seg_num_2;
  wire [3:2]NLW_cnt_ms0_carry__2_CO_UNCONNECTED;
  wire [3:3]NLW_cnt_ms0_carry__2_O_UNCONNECTED;

  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT3 #(
    .INIT(8'hFE)) 
    \AN[0]_i_1 
       (.I0(cnt[1]),
        .I1(cnt[2]),
        .I2(cnt[0]),
        .O(\AN[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT3 #(
    .INIT(8'hEF)) 
    \AN[1]_i_1 
       (.I0(cnt[1]),
        .I1(cnt[2]),
        .I2(cnt[0]),
        .O(\AN[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT3 #(
    .INIT(8'hEF)) 
    \AN[2]_i_1 
       (.I0(cnt[2]),
        .I1(cnt[0]),
        .I2(cnt[1]),
        .O(\AN[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT3 #(
    .INIT(8'hF7)) 
    \AN[3]_i_1 
       (.I0(cnt[1]),
        .I1(cnt[0]),
        .I2(cnt[2]),
        .O(\AN[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT3 #(
    .INIT(8'hEF)) 
    \AN[4]_i_1 
       (.I0(cnt[1]),
        .I1(cnt[0]),
        .I2(cnt[2]),
        .O(\AN[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT3 #(
    .INIT(8'hF7)) 
    \AN[5]_i_1 
       (.I0(cnt[2]),
        .I1(cnt[0]),
        .I2(cnt[1]),
        .O(\AN[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'hF7)) 
    \AN[6]_i_1 
       (.I0(cnt[1]),
        .I1(cnt[2]),
        .I2(cnt[0]),
        .O(\AN[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'h7F)) 
    \AN[7]_i_1 
       (.I0(cnt[2]),
        .I1(cnt[0]),
        .I2(cnt[1]),
        .O(\AN[7]_i_1_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \AN[7]_i_2 
       (.I0(rst_n),
        .O(\AN[7]_i_2_n_0 ));
  FDPE \AN_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\AN[0]_i_1_n_0 ),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(AN[0]));
  FDPE \AN_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\AN[1]_i_1_n_0 ),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(AN[1]));
  FDPE \AN_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\AN[2]_i_1_n_0 ),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(AN[2]));
  FDPE \AN_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .D(\AN[3]_i_1_n_0 ),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(AN[3]));
  FDPE \AN_reg[4] 
       (.C(clk),
        .CE(1'b1),
        .D(\AN[4]_i_1_n_0 ),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(AN[4]));
  FDPE \AN_reg[5] 
       (.C(clk),
        .CE(1'b1),
        .D(\AN[5]_i_1_n_0 ),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(AN[5]));
  FDPE \AN_reg[6] 
       (.C(clk),
        .CE(1'b1),
        .D(\AN[6]_i_1_n_0 ),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(AN[6]));
  FDPE \AN_reg[7] 
       (.C(clk),
        .CE(1'b1),
        .D(\AN[7]_i_1_n_0 ),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(AN[7]));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT2 #(
    .INIT(4'h9)) 
    \cnt[0]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(cnt[0]),
        .O(\cnt[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT3 #(
    .INIT(8'hD2)) 
    \cnt[1]_i_1 
       (.I0(cnt[0]),
        .I1(\cnt_ms[15]_i_2_n_0 ),
        .I2(cnt[1]),
        .O(\cnt[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT4 #(
    .INIT(16'hF708)) 
    \cnt[2]_i_1 
       (.I0(cnt[1]),
        .I1(cnt[0]),
        .I2(\cnt_ms[15]_i_2_n_0 ),
        .I3(cnt[2]),
        .O(\cnt[2]_i_1_n_0 ));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 cnt_ms0_carry
       (.CI(1'b0),
        .CO({cnt_ms0_carry_n_0,cnt_ms0_carry_n_1,cnt_ms0_carry_n_2,cnt_ms0_carry_n_3}),
        .CYINIT(cnt_ms[0]),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(data0[4:1]),
        .S(cnt_ms[4:1]));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 cnt_ms0_carry__0
       (.CI(cnt_ms0_carry_n_0),
        .CO({cnt_ms0_carry__0_n_0,cnt_ms0_carry__0_n_1,cnt_ms0_carry__0_n_2,cnt_ms0_carry__0_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(data0[8:5]),
        .S(cnt_ms[8:5]));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 cnt_ms0_carry__1
       (.CI(cnt_ms0_carry__0_n_0),
        .CO({cnt_ms0_carry__1_n_0,cnt_ms0_carry__1_n_1,cnt_ms0_carry__1_n_2,cnt_ms0_carry__1_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(data0[12:9]),
        .S(cnt_ms[12:9]));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 cnt_ms0_carry__2
       (.CI(cnt_ms0_carry__1_n_0),
        .CO({NLW_cnt_ms0_carry__2_CO_UNCONNECTED[3:2],cnt_ms0_carry__2_n_2,cnt_ms0_carry__2_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({NLW_cnt_ms0_carry__2_O_UNCONNECTED[3],data0[15:13]}),
        .S({1'b0,cnt_ms[15:13]}));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT1 #(
    .INIT(2'h1)) 
    \cnt_ms[0]_i_1 
       (.I0(cnt_ms[0]),
        .O(cnt_ms_1[0]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[10]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[10]),
        .O(cnt_ms_1[10]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[11]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[11]),
        .O(cnt_ms_1[11]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[12]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[12]),
        .O(cnt_ms_1[12]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[13]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[13]),
        .O(cnt_ms_1[13]));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[14]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[14]),
        .O(cnt_ms_1[14]));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[15]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[15]),
        .O(cnt_ms_1[15]));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \cnt_ms[15]_i_2 
       (.I0(\cnt_ms[15]_i_3_n_0 ),
        .I1(\cnt_ms[15]_i_4_n_0 ),
        .I2(\cnt_ms[15]_i_5_n_0 ),
        .I3(\cnt_ms[15]_i_6_n_0 ),
        .O(\cnt_ms[15]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFFEF)) 
    \cnt_ms[15]_i_3 
       (.I0(cnt_ms[5]),
        .I1(cnt_ms[4]),
        .I2(cnt_ms[6]),
        .I3(cnt_ms[7]),
        .O(\cnt_ms[15]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT4 #(
    .INIT(16'h7FFF)) 
    \cnt_ms[15]_i_4 
       (.I0(cnt_ms[1]),
        .I1(cnt_ms[0]),
        .I2(cnt_ms[3]),
        .I3(cnt_ms[2]),
        .O(\cnt_ms[15]_i_4_n_0 ));
  LUT4 #(
    .INIT(16'hEFFF)) 
    \cnt_ms[15]_i_5 
       (.I0(cnt_ms[13]),
        .I1(cnt_ms[12]),
        .I2(cnt_ms[15]),
        .I3(cnt_ms[14]),
        .O(\cnt_ms[15]_i_5_n_0 ));
  LUT4 #(
    .INIT(16'hFFF7)) 
    \cnt_ms[15]_i_6 
       (.I0(cnt_ms[9]),
        .I1(cnt_ms[8]),
        .I2(cnt_ms[11]),
        .I3(cnt_ms[10]),
        .O(\cnt_ms[15]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[1]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[1]),
        .O(cnt_ms_1[1]));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[2]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[2]),
        .O(cnt_ms_1[2]));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[3]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[3]),
        .O(cnt_ms_1[3]));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[4]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[4]),
        .O(cnt_ms_1[4]));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[5]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[5]),
        .O(cnt_ms_1[5]));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[6]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[6]),
        .O(cnt_ms_1[6]));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[7]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[7]),
        .O(cnt_ms_1[7]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[8]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[8]),
        .O(cnt_ms_1[8]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \cnt_ms[9]_i_1 
       (.I0(\cnt_ms[15]_i_2_n_0 ),
        .I1(data0[9]),
        .O(cnt_ms_1[9]));
  FDCE \cnt_ms_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[0]),
        .Q(cnt_ms[0]));
  FDCE \cnt_ms_reg[10] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[10]),
        .Q(cnt_ms[10]));
  FDCE \cnt_ms_reg[11] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[11]),
        .Q(cnt_ms[11]));
  FDCE \cnt_ms_reg[12] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[12]),
        .Q(cnt_ms[12]));
  FDCE \cnt_ms_reg[13] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[13]),
        .Q(cnt_ms[13]));
  FDCE \cnt_ms_reg[14] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[14]),
        .Q(cnt_ms[14]));
  FDCE \cnt_ms_reg[15] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[15]),
        .Q(cnt_ms[15]));
  FDCE \cnt_ms_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[1]),
        .Q(cnt_ms[1]));
  FDCE \cnt_ms_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[2]),
        .Q(cnt_ms[2]));
  FDCE \cnt_ms_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[3]),
        .Q(cnt_ms[3]));
  FDCE \cnt_ms_reg[4] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[4]),
        .Q(cnt_ms[4]));
  FDCE \cnt_ms_reg[5] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[5]),
        .Q(cnt_ms[5]));
  FDCE \cnt_ms_reg[6] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[6]),
        .Q(cnt_ms[6]));
  FDCE \cnt_ms_reg[7] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[7]),
        .Q(cnt_ms[7]));
  FDCE \cnt_ms_reg[8] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[8]),
        .Q(cnt_ms[8]));
  FDCE \cnt_ms_reg[9] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(cnt_ms_1[9]),
        .Q(cnt_ms[9]));
  FDCE \cnt_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(\cnt[0]_i_1_n_0 ),
        .Q(cnt[0]));
  FDCE \cnt_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(\cnt[1]_i_1_n_0 ),
        .Q(cnt[1]));
  FDCE \cnt_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(\cnt[2]_i_1_n_0 ),
        .Q(cnt[2]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT4 #(
    .INIT(16'h2094)) 
    \seg_data_o[0]_i_1 
       (.I0(seg_num[3]),
        .I1(seg_num[2]),
        .I2(seg_num[0]),
        .I3(seg_num[1]),
        .O(seg_data_o_0[0]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT4 #(
    .INIT(16'hA4C8)) 
    \seg_data_o[1]_i_1 
       (.I0(seg_num[3]),
        .I1(seg_num[2]),
        .I2(seg_num[1]),
        .I3(seg_num[0]),
        .O(seg_data_o_0[1]));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT4 #(
    .INIT(16'hA210)) 
    \seg_data_o[2]_i_1 
       (.I0(seg_num[3]),
        .I1(seg_num[0]),
        .I2(seg_num[1]),
        .I3(seg_num[2]),
        .O(seg_data_o_0[2]));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT4 #(
    .INIT(16'hC214)) 
    \seg_data_o[3]_i_1 
       (.I0(seg_num[3]),
        .I1(seg_num[2]),
        .I2(seg_num[0]),
        .I3(seg_num[1]),
        .O(seg_data_o_0[3]));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT4 #(
    .INIT(16'h5710)) 
    \seg_data_o[4]_i_1 
       (.I0(seg_num[3]),
        .I1(seg_num[1]),
        .I2(seg_num[2]),
        .I3(seg_num[0]),
        .O(seg_data_o_0[4]));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT4 #(
    .INIT(16'h5190)) 
    \seg_data_o[5]_i_1 
       (.I0(seg_num[3]),
        .I1(seg_num[2]),
        .I2(seg_num[0]),
        .I3(seg_num[1]),
        .O(seg_data_o_0[5]));
  LUT4 #(
    .INIT(16'h4025)) 
    \seg_data_o[6]_i_1 
       (.I0(seg_num[3]),
        .I1(seg_num[0]),
        .I2(seg_num[2]),
        .I3(seg_num[1]),
        .O(seg_data_o_0[6]));
  FDPE \seg_data_o_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(seg_data_o_0[0]),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(seg_data_o[0]));
  FDPE \seg_data_o_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(seg_data_o_0[1]),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(seg_data_o[1]));
  FDPE \seg_data_o_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(seg_data_o_0[2]),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(seg_data_o[2]));
  FDPE \seg_data_o_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .D(seg_data_o_0[3]),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(seg_data_o[3]));
  FDPE \seg_data_o_reg[4] 
       (.C(clk),
        .CE(1'b1),
        .D(seg_data_o_0[4]),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(seg_data_o[4]));
  FDPE \seg_data_o_reg[5] 
       (.C(clk),
        .CE(1'b1),
        .D(seg_data_o_0[5]),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(seg_data_o[5]));
  FDPE \seg_data_o_reg[6] 
       (.C(clk),
        .CE(1'b1),
        .D(seg_data_o_0[6]),
        .PRE(\AN[7]_i_2_n_0 ),
        .Q(seg_data_o[6]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \seg_num[0]_i_2 
       (.I0(Data_i[12]),
        .I1(Data_i[8]),
        .I2(cnt[1]),
        .I3(Data_i[4]),
        .I4(cnt[0]),
        .I5(Data_i[0]),
        .O(\seg_num[0]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \seg_num[0]_i_3 
       (.I0(Data_i[28]),
        .I1(Data_i[24]),
        .I2(cnt[1]),
        .I3(Data_i[20]),
        .I4(cnt[0]),
        .I5(Data_i[16]),
        .O(\seg_num[0]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \seg_num[1]_i_2 
       (.I0(Data_i[13]),
        .I1(Data_i[9]),
        .I2(cnt[1]),
        .I3(Data_i[5]),
        .I4(cnt[0]),
        .I5(Data_i[1]),
        .O(\seg_num[1]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \seg_num[1]_i_3 
       (.I0(Data_i[29]),
        .I1(Data_i[25]),
        .I2(cnt[1]),
        .I3(Data_i[21]),
        .I4(cnt[0]),
        .I5(Data_i[17]),
        .O(\seg_num[1]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \seg_num[2]_i_2 
       (.I0(Data_i[14]),
        .I1(Data_i[10]),
        .I2(cnt[1]),
        .I3(Data_i[6]),
        .I4(cnt[0]),
        .I5(Data_i[2]),
        .O(\seg_num[2]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \seg_num[2]_i_3 
       (.I0(Data_i[30]),
        .I1(Data_i[26]),
        .I2(cnt[1]),
        .I3(Data_i[22]),
        .I4(cnt[0]),
        .I5(Data_i[18]),
        .O(\seg_num[2]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \seg_num[3]_i_2 
       (.I0(Data_i[15]),
        .I1(Data_i[11]),
        .I2(cnt[1]),
        .I3(Data_i[7]),
        .I4(cnt[0]),
        .I5(Data_i[3]),
        .O(\seg_num[3]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \seg_num[3]_i_3 
       (.I0(Data_i[31]),
        .I1(Data_i[27]),
        .I2(cnt[1]),
        .I3(Data_i[23]),
        .I4(cnt[0]),
        .I5(Data_i[19]),
        .O(\seg_num[3]_i_3_n_0 ));
  FDCE \seg_num_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(seg_num_2[0]),
        .Q(seg_num[0]));
  MUXF7 \seg_num_reg[0]_i_1 
       (.I0(\seg_num[0]_i_2_n_0 ),
        .I1(\seg_num[0]_i_3_n_0 ),
        .O(seg_num_2[0]),
        .S(cnt[2]));
  FDCE \seg_num_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(seg_num_2[1]),
        .Q(seg_num[1]));
  MUXF7 \seg_num_reg[1]_i_1 
       (.I0(\seg_num[1]_i_2_n_0 ),
        .I1(\seg_num[1]_i_3_n_0 ),
        .O(seg_num_2[1]),
        .S(cnt[2]));
  FDCE \seg_num_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(seg_num_2[2]),
        .Q(seg_num[2]));
  MUXF7 \seg_num_reg[2]_i_1 
       (.I0(\seg_num[2]_i_2_n_0 ),
        .I1(\seg_num[2]_i_3_n_0 ),
        .O(seg_num_2[2]),
        .S(cnt[2]));
  FDCE \seg_num_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\AN[7]_i_2_n_0 ),
        .D(seg_num_2[3]),
        .Q(seg_num[3]));
  MUXF7 \seg_num_reg[3]_i_1 
       (.I0(\seg_num[3]_i_2_n_0 ),
        .I1(\seg_num[3]_i_3_n_0 ),
        .O(seg_num_2[3]),
        .S(cnt[2]));
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
