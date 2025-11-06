// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Tue Oct 29 12:52:06 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim
//               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_pc_0_0/risc32_pc_0_0_sim_netlist.v
// Design      : risc32_pc_0_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CHECK_LICENSE_TYPE = "risc32_pc_0_0,pc,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "pc,Vivado 2022.2" *) 
(* NotValidForBitStream *)
module risc32_pc_0_0
   (clk,
    pc_clr,
    data_i,
    q);
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk, FREQ_HZ 100000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN risc32_clk, INSERT_VIP 0" *) input clk;
  input pc_clr;
  input [31:0]data_i;
  output [31:0]q;

  wire clk;
  wire [31:0]data_i;
  wire pc_clr;
  wire [31:0]q;

  risc32_pc_0_0_pc inst
       (.clk(clk),
        .data_i(data_i),
        .pc_clr(pc_clr),
        .q(q));
endmodule

(* ORIG_REF_NAME = "pc" *) 
module risc32_pc_0_0_pc
   (q,
    data_i,
    clk,
    pc_clr);
  output [31:0]q;
  input [31:0]data_i;
  input clk;
  input pc_clr;

  wire clk;
  wire [31:0]data_i;
  wire pc_clr;
  wire [31:0]q;
  wire \q[31]_i_1_n_0 ;

  LUT1 #(
    .INIT(2'h1)) 
    \q[31]_i_1 
       (.I0(pc_clr),
        .O(\q[31]_i_1_n_0 ));
  FDCE \q_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[0]),
        .Q(q[0]));
  FDCE \q_reg[10] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[10]),
        .Q(q[10]));
  FDCE \q_reg[11] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[11]),
        .Q(q[11]));
  FDCE \q_reg[12] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[12]),
        .Q(q[12]));
  FDCE \q_reg[13] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[13]),
        .Q(q[13]));
  FDCE \q_reg[14] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[14]),
        .Q(q[14]));
  FDCE \q_reg[15] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[15]),
        .Q(q[15]));
  FDCE \q_reg[16] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[16]),
        .Q(q[16]));
  FDCE \q_reg[17] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[17]),
        .Q(q[17]));
  FDCE \q_reg[18] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[18]),
        .Q(q[18]));
  FDCE \q_reg[19] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[19]),
        .Q(q[19]));
  FDCE \q_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[1]),
        .Q(q[1]));
  FDCE \q_reg[20] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[20]),
        .Q(q[20]));
  FDCE \q_reg[21] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[21]),
        .Q(q[21]));
  FDCE \q_reg[22] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[22]),
        .Q(q[22]));
  FDCE \q_reg[23] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[23]),
        .Q(q[23]));
  FDCE \q_reg[24] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[24]),
        .Q(q[24]));
  FDCE \q_reg[25] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[25]),
        .Q(q[25]));
  FDCE \q_reg[26] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[26]),
        .Q(q[26]));
  FDCE \q_reg[27] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[27]),
        .Q(q[27]));
  FDCE \q_reg[28] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[28]),
        .Q(q[28]));
  FDCE \q_reg[29] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[29]),
        .Q(q[29]));
  FDCE \q_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[2]),
        .Q(q[2]));
  FDCE \q_reg[30] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[30]),
        .Q(q[30]));
  FDCE \q_reg[31] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[31]),
        .Q(q[31]));
  FDCE \q_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[3]),
        .Q(q[3]));
  FDCE \q_reg[4] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[4]),
        .Q(q[4]));
  FDCE \q_reg[5] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[5]),
        .Q(q[5]));
  FDCE \q_reg[6] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[6]),
        .Q(q[6]));
  FDCE \q_reg[7] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[7]),
        .Q(q[7]));
  FDCE \q_reg[8] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[8]),
        .Q(q[8]));
  FDCE \q_reg[9] 
       (.C(clk),
        .CE(1'b1),
        .CLR(\q[31]_i_1_n_0 ),
        .D(data_i[9]),
        .Q(q[9]));
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
