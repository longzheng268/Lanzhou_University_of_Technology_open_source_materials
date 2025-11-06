// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Mon Oct 28 16:05:23 2024
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
   (inst,
    inst_addr);
  output [9:0]inst;
  input [5:0]inst_addr;

  wire [9:0]inst;
  wire [5:0]inst_addr;

  LUT6 #(
    .INIT(64'h0100000000000010)) 
    \inst[10]_INST_0 
       (.I0(inst_addr[1]),
        .I1(inst_addr[0]),
        .I2(inst_addr[5]),
        .I3(inst_addr[4]),
        .I4(inst_addr[3]),
        .I5(inst_addr[2]),
        .O(inst[2]));
  LUT5 #(
    .INIT(32'h00000002)) 
    \inst[15]_INST_0 
       (.I0(inst_addr[5]),
        .I1(inst_addr[3]),
        .I2(inst_addr[1]),
        .I3(inst_addr[4]),
        .I4(inst_addr[0]),
        .O(inst[4]));
  LUT6 #(
    .INIT(64'h0001000100011001)) 
    \inst[20]_INST_0 
       (.I0(inst_addr[0]),
        .I1(inst_addr[1]),
        .I2(inst_addr[5]),
        .I3(inst_addr[2]),
        .I4(inst_addr[4]),
        .I5(inst_addr[3]),
        .O(inst[5]));
  LUT6 #(
    .INIT(64'h0001000300000300)) 
    \inst[21]_INST_0 
       (.I0(inst_addr[4]),
        .I1(inst_addr[1]),
        .I2(inst_addr[0]),
        .I3(inst_addr[3]),
        .I4(inst_addr[5]),
        .I5(inst_addr[2]),
        .O(inst[6]));
  LUT6 #(
    .INIT(64'h0000100000100000)) 
    \inst[23]_INST_0 
       (.I0(inst_addr[1]),
        .I1(inst_addr[0]),
        .I2(inst_addr[2]),
        .I3(inst_addr[4]),
        .I4(inst_addr[5]),
        .I5(inst_addr[3]),
        .O(inst[7]));
  LUT6 #(
    .INIT(64'h0000000000000100)) 
    \inst[24]_INST_0 
       (.I0(inst_addr[0]),
        .I1(inst_addr[4]),
        .I2(inst_addr[1]),
        .I3(inst_addr[5]),
        .I4(inst_addr[3]),
        .I5(inst_addr[2]),
        .O(inst[8]));
  LUT6 #(
    .INIT(64'h0000000000000001)) 
    \inst[31]_INST_0 
       (.I0(inst_addr[5]),
        .I1(inst_addr[4]),
        .I2(inst_addr[1]),
        .I3(inst_addr[0]),
        .I4(inst_addr[3]),
        .I5(inst_addr[2]),
        .O(inst[9]));
  LUT6 #(
    .INIT(64'h0000000000001000)) 
    \inst[5]_INST_0 
       (.I0(inst_addr[1]),
        .I1(inst_addr[4]),
        .I2(inst_addr[5]),
        .I3(inst_addr[2]),
        .I4(inst_addr[3]),
        .I5(inst_addr[0]),
        .O(inst[3]));
  LUT6 #(
    .INIT(64'h0000000001010111)) 
    \inst[7]_INST_0 
       (.I0(inst_addr[0]),
        .I1(inst_addr[2]),
        .I2(inst_addr[5]),
        .I3(inst_addr[3]),
        .I4(inst_addr[4]),
        .I5(inst_addr[1]),
        .O(inst[0]));
  LUT6 #(
    .INIT(64'h0000010101010010)) 
    \inst[8]_INST_0 
       (.I0(inst_addr[0]),
        .I1(inst_addr[1]),
        .I2(inst_addr[5]),
        .I3(inst_addr[4]),
        .I4(inst_addr[3]),
        .I5(inst_addr[2]),
        .O(inst[1]));
endmodule

(* CHECK_LICENSE_TYPE = "risc32_imem_0_0,imem,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "imem,Vivado 2022.2" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (inst_addr,
    inst);
  input [31:0]inst_addr;
  output [31:0]inst;

  wire \<const0> ;
  wire [31:4]\^inst ;
  wire [31:0]inst_addr;

  assign inst[31] = \^inst [31];
  assign inst[30] = \<const0> ;
  assign inst[29] = \<const0> ;
  assign inst[28] = \<const0> ;
  assign inst[27] = \<const0> ;
  assign inst[26] = \<const0> ;
  assign inst[25] = \<const0> ;
  assign inst[24:20] = \^inst [24:20];
  assign inst[19] = \<const0> ;
  assign inst[18] = \<const0> ;
  assign inst[17] = \<const0> ;
  assign inst[16] = \<const0> ;
  assign inst[15] = \^inst [15];
  assign inst[14] = \<const0> ;
  assign inst[13] = \<const0> ;
  assign inst[12] = \<const0> ;
  assign inst[11:7] = \^inst [11:7];
  assign inst[6] = \<const0> ;
  assign inst[5] = \^inst [11];
  assign inst[4] = \^inst [4];
  assign inst[3] = \<const0> ;
  assign inst[2] = \<const0> ;
  assign inst[1] = \^inst [4];
  assign inst[0] = \^inst [4];
  GND GND
       (.G(\<const0> ));
  LUT5 #(
    .INIT(32'h0000001F)) 
    \inst[0]_INST_0 
       (.I0(inst_addr[4]),
        .I1(inst_addr[3]),
        .I2(inst_addr[5]),
        .I3(inst_addr[0]),
        .I4(inst_addr[1]),
        .O(\^inst [4]));
  LUT6 #(
    .INIT(64'h0000000000001464)) 
    \inst[22]_INST_0 
       (.I0(inst_addr[5]),
        .I1(inst_addr[4]),
        .I2(inst_addr[2]),
        .I3(inst_addr[3]),
        .I4(inst_addr[0]),
        .I5(inst_addr[1]),
        .O(\^inst [22]));
  LUT6 #(
    .INIT(64'h0000000000001324)) 
    \inst[9]_INST_0 
       (.I0(inst_addr[3]),
        .I1(inst_addr[5]),
        .I2(inst_addr[2]),
        .I3(inst_addr[4]),
        .I4(inst_addr[0]),
        .I5(inst_addr[1]),
        .O(\^inst [9]));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_imem inst__0
       (.inst({\^inst [31],\^inst [24:23],\^inst [21:20],\^inst [15],\^inst [11:10],\^inst [8:7]}),
        .inst_addr(inst_addr[5:0]));
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
