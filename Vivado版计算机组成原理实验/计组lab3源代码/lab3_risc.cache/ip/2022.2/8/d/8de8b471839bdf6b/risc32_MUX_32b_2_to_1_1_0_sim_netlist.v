// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Mon Oct 28 16:04:11 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_MUX_32b_2_to_1_1_0_sim_netlist.v
// Design      : risc32_MUX_32b_2_to_1_1_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_MUX_32b_2_to_1
   (result,
    data1,
    data0,
    sel);
  output [31:0]result;
  input [31:0]data1;
  input [31:0]data0;
  input sel;

  wire [31:0]data0;
  wire [31:0]data1;
  wire [31:0]result;
  wire sel;

  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[0]_INST_0 
       (.I0(data1[0]),
        .I1(data0[0]),
        .I2(sel),
        .O(result[0]));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[10]_INST_0 
       (.I0(data1[10]),
        .I1(data0[10]),
        .I2(sel),
        .O(result[10]));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[11]_INST_0 
       (.I0(data1[11]),
        .I1(data0[11]),
        .I2(sel),
        .O(result[11]));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[12]_INST_0 
       (.I0(data1[12]),
        .I1(data0[12]),
        .I2(sel),
        .O(result[12]));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[13]_INST_0 
       (.I0(data1[13]),
        .I1(data0[13]),
        .I2(sel),
        .O(result[13]));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[14]_INST_0 
       (.I0(data1[14]),
        .I1(data0[14]),
        .I2(sel),
        .O(result[14]));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[15]_INST_0 
       (.I0(data1[15]),
        .I1(data0[15]),
        .I2(sel),
        .O(result[15]));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[16]_INST_0 
       (.I0(data1[16]),
        .I1(data0[16]),
        .I2(sel),
        .O(result[16]));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[17]_INST_0 
       (.I0(data1[17]),
        .I1(data0[17]),
        .I2(sel),
        .O(result[17]));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[18]_INST_0 
       (.I0(data1[18]),
        .I1(data0[18]),
        .I2(sel),
        .O(result[18]));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[19]_INST_0 
       (.I0(data1[19]),
        .I1(data0[19]),
        .I2(sel),
        .O(result[19]));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[1]_INST_0 
       (.I0(data1[1]),
        .I1(data0[1]),
        .I2(sel),
        .O(result[1]));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[20]_INST_0 
       (.I0(data1[20]),
        .I1(data0[20]),
        .I2(sel),
        .O(result[20]));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[21]_INST_0 
       (.I0(data1[21]),
        .I1(data0[21]),
        .I2(sel),
        .O(result[21]));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[22]_INST_0 
       (.I0(data1[22]),
        .I1(data0[22]),
        .I2(sel),
        .O(result[22]));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[23]_INST_0 
       (.I0(data1[23]),
        .I1(data0[23]),
        .I2(sel),
        .O(result[23]));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[24]_INST_0 
       (.I0(data1[24]),
        .I1(data0[24]),
        .I2(sel),
        .O(result[24]));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[25]_INST_0 
       (.I0(data1[25]),
        .I1(data0[25]),
        .I2(sel),
        .O(result[25]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[26]_INST_0 
       (.I0(data1[26]),
        .I1(data0[26]),
        .I2(sel),
        .O(result[26]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[27]_INST_0 
       (.I0(data1[27]),
        .I1(data0[27]),
        .I2(sel),
        .O(result[27]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[28]_INST_0 
       (.I0(data1[28]),
        .I1(data0[28]),
        .I2(sel),
        .O(result[28]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[29]_INST_0 
       (.I0(data1[29]),
        .I1(data0[29]),
        .I2(sel),
        .O(result[29]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[2]_INST_0 
       (.I0(data1[2]),
        .I1(data0[2]),
        .I2(sel),
        .O(result[2]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[30]_INST_0 
       (.I0(data1[30]),
        .I1(data0[30]),
        .I2(sel),
        .O(result[30]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[31]_INST_0 
       (.I0(data1[31]),
        .I1(data0[31]),
        .I2(sel),
        .O(result[31]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[3]_INST_0 
       (.I0(data1[3]),
        .I1(data0[3]),
        .I2(sel),
        .O(result[3]));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[4]_INST_0 
       (.I0(data1[4]),
        .I1(data0[4]),
        .I2(sel),
        .O(result[4]));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[5]_INST_0 
       (.I0(data1[5]),
        .I1(data0[5]),
        .I2(sel),
        .O(result[5]));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[6]_INST_0 
       (.I0(data1[6]),
        .I1(data0[6]),
        .I2(sel),
        .O(result[6]));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[7]_INST_0 
       (.I0(data1[7]),
        .I1(data0[7]),
        .I2(sel),
        .O(result[7]));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[8]_INST_0 
       (.I0(data1[8]),
        .I1(data0[8]),
        .I2(sel),
        .O(result[8]));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \result[9]_INST_0 
       (.I0(data1[9]),
        .I1(data0[9]),
        .I2(sel),
        .O(result[9]));
endmodule

(* CHECK_LICENSE_TYPE = "risc32_MUX_32b_2_to_1_1_0,MUX_32b_2_to_1,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "MUX_32b_2_to_1,Vivado 2022.2" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (data0,
    data1,
    sel,
    result);
  input [31:0]data0;
  input [31:0]data1;
  input sel;
  output [31:0]result;

  wire [31:0]data0;
  wire [31:0]data1;
  wire [31:0]result;
  wire sel;

  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_MUX_32b_2_to_1 inst
       (.data0(data0),
        .data1(data1),
        .result(result),
        .sel(sel));
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
