// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Tue Oct 29 10:56:53 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_imem_0_0_sim_netlist.v
// Design      : risc32_imem_0_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_imem
   (inst_o,
    inst_addr);
  output [9:0]inst_o;
  input [5:0]inst_addr;

  wire [5:0]inst_addr;
  wire [9:0]inst_o;

  LUT6 #(
    .INIT(64'h0100000000000010)) 
    \inst_o[10]_INST_0 
       (.I0(inst_addr[1]),
        .I1(inst_addr[0]),
        .I2(inst_addr[5]),
        .I3(inst_addr[4]),
        .I4(inst_addr[3]),
        .I5(inst_addr[2]),
        .O(inst_o[2]));
  LUT5 #(
    .INIT(32'h00000002)) 
    \inst_o[15]_INST_0 
       (.I0(inst_addr[5]),
        .I1(inst_addr[3]),
        .I2(inst_addr[1]),
        .I3(inst_addr[4]),
        .I4(inst_addr[0]),
        .O(inst_o[4]));
  LUT6 #(
    .INIT(64'h0001000100011001)) 
    \inst_o[20]_INST_0 
       (.I0(inst_addr[0]),
        .I1(inst_addr[1]),
        .I2(inst_addr[5]),
        .I3(inst_addr[2]),
        .I4(inst_addr[4]),
        .I5(inst_addr[3]),
        .O(inst_o[5]));
  LUT6 #(
    .INIT(64'h0001000300000300)) 
    \inst_o[21]_INST_0 
       (.I0(inst_addr[4]),
        .I1(inst_addr[1]),
        .I2(inst_addr[0]),
        .I3(inst_addr[3]),
        .I4(inst_addr[5]),
        .I5(inst_addr[2]),
        .O(inst_o[6]));
  LUT6 #(
    .INIT(64'h0000100000100000)) 
    \inst_o[23]_INST_0 
       (.I0(inst_addr[1]),
        .I1(inst_addr[0]),
        .I2(inst_addr[2]),
        .I3(inst_addr[4]),
        .I4(inst_addr[5]),
        .I5(inst_addr[3]),
        .O(inst_o[7]));
  LUT6 #(
    .INIT(64'h0000000000000100)) 
    \inst_o[24]_INST_0 
       (.I0(inst_addr[0]),
        .I1(inst_addr[4]),
        .I2(inst_addr[1]),
        .I3(inst_addr[5]),
        .I4(inst_addr[3]),
        .I5(inst_addr[2]),
        .O(inst_o[8]));
  LUT6 #(
    .INIT(64'h0000000000000001)) 
    \inst_o[31]_INST_0 
       (.I0(inst_addr[5]),
        .I1(inst_addr[4]),
        .I2(inst_addr[1]),
        .I3(inst_addr[0]),
        .I4(inst_addr[3]),
        .I5(inst_addr[2]),
        .O(inst_o[9]));
  LUT6 #(
    .INIT(64'h0000000000001000)) 
    \inst_o[5]_INST_0 
       (.I0(inst_addr[1]),
        .I1(inst_addr[4]),
        .I2(inst_addr[5]),
        .I3(inst_addr[2]),
        .I4(inst_addr[3]),
        .I5(inst_addr[0]),
        .O(inst_o[3]));
  LUT6 #(
    .INIT(64'h0000000001010111)) 
    \inst_o[7]_INST_0 
       (.I0(inst_addr[0]),
        .I1(inst_addr[2]),
        .I2(inst_addr[5]),
        .I3(inst_addr[3]),
        .I4(inst_addr[4]),
        .I5(inst_addr[1]),
        .O(inst_o[0]));
  LUT6 #(
    .INIT(64'h0000010101010010)) 
    \inst_o[8]_INST_0 
       (.I0(inst_addr[0]),
        .I1(inst_addr[1]),
        .I2(inst_addr[5]),
        .I3(inst_addr[4]),
        .I4(inst_addr[3]),
        .I5(inst_addr[2]),
        .O(inst_o[1]));
endmodule

(* CHECK_LICENSE_TYPE = "risc32_imem_0_0,imem,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "imem,Vivado 2022.2" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (inst_addr,
    inst_o);
  input [31:0]inst_addr;
  output [31:0]inst_o;

  wire \<const0> ;
  wire [31:0]inst_addr;
  wire [31:4]\^inst_o ;

  assign inst_o[31] = \^inst_o [31];
  assign inst_o[30] = \<const0> ;
  assign inst_o[29] = \<const0> ;
  assign inst_o[28] = \<const0> ;
  assign inst_o[27] = \<const0> ;
  assign inst_o[26] = \<const0> ;
  assign inst_o[25] = \<const0> ;
  assign inst_o[24:20] = \^inst_o [24:20];
  assign inst_o[19] = \<const0> ;
  assign inst_o[18] = \<const0> ;
  assign inst_o[17] = \<const0> ;
  assign inst_o[16] = \<const0> ;
  assign inst_o[15] = \^inst_o [15];
  assign inst_o[14] = \<const0> ;
  assign inst_o[13] = \<const0> ;
  assign inst_o[12] = \<const0> ;
  assign inst_o[11:7] = \^inst_o [11:7];
  assign inst_o[6] = \<const0> ;
  assign inst_o[5] = \^inst_o [11];
  assign inst_o[4] = \^inst_o [4];
  assign inst_o[3] = \<const0> ;
  assign inst_o[2] = \<const0> ;
  assign inst_o[1] = \^inst_o [4];
  assign inst_o[0] = \^inst_o [4];
  GND GND
       (.G(\<const0> ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_imem inst
       (.inst_addr(inst_addr[5:0]),
        .inst_o({\^inst_o [31],\^inst_o [24:23],\^inst_o [21:20],\^inst_o [15],\^inst_o [11:10],\^inst_o [8:7]}));
  LUT5 #(
    .INIT(32'h0000001F)) 
    \inst_o[0]_INST_0 
       (.I0(inst_addr[4]),
        .I1(inst_addr[3]),
        .I2(inst_addr[5]),
        .I3(inst_addr[0]),
        .I4(inst_addr[1]),
        .O(\^inst_o [4]));
  LUT6 #(
    .INIT(64'h0000000000001464)) 
    \inst_o[22]_INST_0 
       (.I0(inst_addr[5]),
        .I1(inst_addr[4]),
        .I2(inst_addr[2]),
        .I3(inst_addr[3]),
        .I4(inst_addr[0]),
        .I5(inst_addr[1]),
        .O(\^inst_o [22]));
  LUT6 #(
    .INIT(64'h0000000000001324)) 
    \inst_o[9]_INST_0 
       (.I0(inst_addr[3]),
        .I1(inst_addr[5]),
        .I2(inst_addr[2]),
        .I3(inst_addr[4]),
        .I4(inst_addr[0]),
        .I5(inst_addr[1]),
        .O(\^inst_o [9]));
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
