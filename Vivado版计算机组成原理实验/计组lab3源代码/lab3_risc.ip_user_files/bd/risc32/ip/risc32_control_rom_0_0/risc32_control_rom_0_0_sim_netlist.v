// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Tue Oct 29 10:56:53 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim
//               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_control_rom_0_0/risc32_control_rom_0_0_sim_netlist.v
// Design      : risc32_control_rom_0_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CHECK_LICENSE_TYPE = "risc32_control_rom_0_0,control_rom,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "control_rom,Vivado 2022.2" *) 
(* NotValidForBitStream *)
module risc32_control_rom_0_0
   (inst_ctr,
    imm_sel,
    bsel,
    mem_sel,
    alusel,
    w_ena,
    mem_ena);
  input [8:0]inst_ctr;
  output imm_sel;
  output bsel;
  output mem_sel;
  output [2:0]alusel;
  output w_ena;
  output mem_ena;

  wire \<const0> ;
  wire bsel;
  wire imm_sel;
  wire [8:0]inst_ctr;
  wire mem_sel;
  wire w_ena;

  assign alusel[2] = \<const0> ;
  assign alusel[1] = \<const0> ;
  assign alusel[0] = \<const0> ;
  assign mem_ena = imm_sel;
  GND GND
       (.G(\<const0> ));
  risc32_control_rom_0_0_control_rom inst
       (.bsel(bsel),
        .imm_sel(imm_sel),
        .inst_ctr(inst_ctr),
        .mem_sel(mem_sel),
        .w_ena(w_ena));
endmodule

(* ORIG_REF_NAME = "control_rom" *) 
module risc32_control_rom_0_0_control_rom
   (w_ena,
    bsel,
    mem_sel,
    imm_sel,
    inst_ctr);
  output w_ena;
  output bsel;
  output mem_sel;
  output imm_sel;
  input [8:0]inst_ctr;

  wire bsel;
  wire imm_sel;
  wire [8:0]inst_ctr;
  wire mem_ena_INST_0_i_1_n_0;
  wire mem_sel;
  wire w_ena;

  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'h04003400)) 
    bsel_INST_0
       (.I0(inst_ctr[8]),
        .I1(inst_ctr[6]),
        .I2(inst_ctr[2]),
        .I3(mem_ena_INST_0_i_1_n_0),
        .I4(inst_ctr[3]),
        .O(bsel));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'h00200000)) 
    mem_ena_INST_0
       (.I0(inst_ctr[3]),
        .I1(inst_ctr[2]),
        .I2(mem_ena_INST_0_i_1_n_0),
        .I3(inst_ctr[8]),
        .I4(inst_ctr[6]),
        .O(imm_sel));
  LUT5 #(
    .INIT(32'h00000001)) 
    mem_ena_INST_0_i_1
       (.I0(inst_ctr[0]),
        .I1(inst_ctr[7]),
        .I2(inst_ctr[4]),
        .I3(inst_ctr[1]),
        .I4(inst_ctr[5]),
        .O(mem_ena_INST_0_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'h00100000)) 
    mem_sel_INST_0
       (.I0(inst_ctr[3]),
        .I1(inst_ctr[2]),
        .I2(mem_ena_INST_0_i_1_n_0),
        .I3(inst_ctr[8]),
        .I4(inst_ctr[6]),
        .O(mem_sel));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'h15000200)) 
    w_ena_INST_0
       (.I0(inst_ctr[6]),
        .I1(inst_ctr[8]),
        .I2(inst_ctr[3]),
        .I3(mem_ena_INST_0_i_1_n_0),
        .I4(inst_ctr[2]),
        .O(w_ena));
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
