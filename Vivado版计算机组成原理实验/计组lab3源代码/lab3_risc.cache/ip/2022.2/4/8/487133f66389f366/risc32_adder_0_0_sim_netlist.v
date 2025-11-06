// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
// Date        : Mon Oct 28 16:04:52 2024
// Host        : cop running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_adder_0_0_sim_netlist.v
// Design      : risc32_adder_0_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7a100tcsg324-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_adder
   (add_out,
    add_a,
    add_b);
  output [31:0]add_out;
  input [31:0]add_a;
  input [31:0]add_b;

  wire [31:0]add_a;
  wire [31:0]add_b;
  wire [31:0]add_out;
  wire \add_out[0]_INST_0_i_1_n_0 ;
  wire \add_out[0]_INST_0_i_2_n_0 ;
  wire \add_out[0]_INST_0_i_3_n_0 ;
  wire \add_out[0]_INST_0_i_4_n_0 ;
  wire \add_out[0]_INST_0_n_0 ;
  wire \add_out[0]_INST_0_n_1 ;
  wire \add_out[0]_INST_0_n_2 ;
  wire \add_out[0]_INST_0_n_3 ;
  wire \add_out[12]_INST_0_i_1_n_0 ;
  wire \add_out[12]_INST_0_i_2_n_0 ;
  wire \add_out[12]_INST_0_i_3_n_0 ;
  wire \add_out[12]_INST_0_i_4_n_0 ;
  wire \add_out[12]_INST_0_n_0 ;
  wire \add_out[12]_INST_0_n_1 ;
  wire \add_out[12]_INST_0_n_2 ;
  wire \add_out[12]_INST_0_n_3 ;
  wire \add_out[16]_INST_0_i_1_n_0 ;
  wire \add_out[16]_INST_0_i_2_n_0 ;
  wire \add_out[16]_INST_0_i_3_n_0 ;
  wire \add_out[16]_INST_0_i_4_n_0 ;
  wire \add_out[16]_INST_0_n_0 ;
  wire \add_out[16]_INST_0_n_1 ;
  wire \add_out[16]_INST_0_n_2 ;
  wire \add_out[16]_INST_0_n_3 ;
  wire \add_out[20]_INST_0_i_1_n_0 ;
  wire \add_out[20]_INST_0_i_2_n_0 ;
  wire \add_out[20]_INST_0_i_3_n_0 ;
  wire \add_out[20]_INST_0_i_4_n_0 ;
  wire \add_out[20]_INST_0_n_0 ;
  wire \add_out[20]_INST_0_n_1 ;
  wire \add_out[20]_INST_0_n_2 ;
  wire \add_out[20]_INST_0_n_3 ;
  wire \add_out[24]_INST_0_i_1_n_0 ;
  wire \add_out[24]_INST_0_i_2_n_0 ;
  wire \add_out[24]_INST_0_i_3_n_0 ;
  wire \add_out[24]_INST_0_i_4_n_0 ;
  wire \add_out[24]_INST_0_n_0 ;
  wire \add_out[24]_INST_0_n_1 ;
  wire \add_out[24]_INST_0_n_2 ;
  wire \add_out[24]_INST_0_n_3 ;
  wire \add_out[28]_INST_0_i_1_n_0 ;
  wire \add_out[28]_INST_0_i_2_n_0 ;
  wire \add_out[28]_INST_0_i_3_n_0 ;
  wire \add_out[28]_INST_0_i_4_n_0 ;
  wire \add_out[28]_INST_0_n_1 ;
  wire \add_out[28]_INST_0_n_2 ;
  wire \add_out[28]_INST_0_n_3 ;
  wire \add_out[4]_INST_0_i_1_n_0 ;
  wire \add_out[4]_INST_0_i_2_n_0 ;
  wire \add_out[4]_INST_0_i_3_n_0 ;
  wire \add_out[4]_INST_0_i_4_n_0 ;
  wire \add_out[4]_INST_0_n_0 ;
  wire \add_out[4]_INST_0_n_1 ;
  wire \add_out[4]_INST_0_n_2 ;
  wire \add_out[4]_INST_0_n_3 ;
  wire \add_out[8]_INST_0_i_1_n_0 ;
  wire \add_out[8]_INST_0_i_2_n_0 ;
  wire \add_out[8]_INST_0_i_3_n_0 ;
  wire \add_out[8]_INST_0_i_4_n_0 ;
  wire \add_out[8]_INST_0_n_0 ;
  wire \add_out[8]_INST_0_n_1 ;
  wire \add_out[8]_INST_0_n_2 ;
  wire \add_out[8]_INST_0_n_3 ;
  wire [3:3]\NLW_add_out[28]_INST_0_CO_UNCONNECTED ;

  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \add_out[0]_INST_0 
       (.CI(1'b0),
        .CO({\add_out[0]_INST_0_n_0 ,\add_out[0]_INST_0_n_1 ,\add_out[0]_INST_0_n_2 ,\add_out[0]_INST_0_n_3 }),
        .CYINIT(1'b0),
        .DI(add_a[3:0]),
        .O(add_out[3:0]),
        .S({\add_out[0]_INST_0_i_1_n_0 ,\add_out[0]_INST_0_i_2_n_0 ,\add_out[0]_INST_0_i_3_n_0 ,\add_out[0]_INST_0_i_4_n_0 }));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[0]_INST_0_i_1 
       (.I0(add_a[3]),
        .I1(add_b[3]),
        .O(\add_out[0]_INST_0_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[0]_INST_0_i_2 
       (.I0(add_a[2]),
        .I1(add_b[2]),
        .O(\add_out[0]_INST_0_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[0]_INST_0_i_3 
       (.I0(add_a[1]),
        .I1(add_b[1]),
        .O(\add_out[0]_INST_0_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[0]_INST_0_i_4 
       (.I0(add_a[0]),
        .I1(add_b[0]),
        .O(\add_out[0]_INST_0_i_4_n_0 ));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \add_out[12]_INST_0 
       (.CI(\add_out[8]_INST_0_n_0 ),
        .CO({\add_out[12]_INST_0_n_0 ,\add_out[12]_INST_0_n_1 ,\add_out[12]_INST_0_n_2 ,\add_out[12]_INST_0_n_3 }),
        .CYINIT(1'b0),
        .DI(add_a[15:12]),
        .O(add_out[15:12]),
        .S({\add_out[12]_INST_0_i_1_n_0 ,\add_out[12]_INST_0_i_2_n_0 ,\add_out[12]_INST_0_i_3_n_0 ,\add_out[12]_INST_0_i_4_n_0 }));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[12]_INST_0_i_1 
       (.I0(add_a[15]),
        .I1(add_b[15]),
        .O(\add_out[12]_INST_0_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[12]_INST_0_i_2 
       (.I0(add_a[14]),
        .I1(add_b[14]),
        .O(\add_out[12]_INST_0_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[12]_INST_0_i_3 
       (.I0(add_a[13]),
        .I1(add_b[13]),
        .O(\add_out[12]_INST_0_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[12]_INST_0_i_4 
       (.I0(add_a[12]),
        .I1(add_b[12]),
        .O(\add_out[12]_INST_0_i_4_n_0 ));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \add_out[16]_INST_0 
       (.CI(\add_out[12]_INST_0_n_0 ),
        .CO({\add_out[16]_INST_0_n_0 ,\add_out[16]_INST_0_n_1 ,\add_out[16]_INST_0_n_2 ,\add_out[16]_INST_0_n_3 }),
        .CYINIT(1'b0),
        .DI(add_a[19:16]),
        .O(add_out[19:16]),
        .S({\add_out[16]_INST_0_i_1_n_0 ,\add_out[16]_INST_0_i_2_n_0 ,\add_out[16]_INST_0_i_3_n_0 ,\add_out[16]_INST_0_i_4_n_0 }));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[16]_INST_0_i_1 
       (.I0(add_a[19]),
        .I1(add_b[19]),
        .O(\add_out[16]_INST_0_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[16]_INST_0_i_2 
       (.I0(add_a[18]),
        .I1(add_b[18]),
        .O(\add_out[16]_INST_0_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[16]_INST_0_i_3 
       (.I0(add_a[17]),
        .I1(add_b[17]),
        .O(\add_out[16]_INST_0_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[16]_INST_0_i_4 
       (.I0(add_a[16]),
        .I1(add_b[16]),
        .O(\add_out[16]_INST_0_i_4_n_0 ));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \add_out[20]_INST_0 
       (.CI(\add_out[16]_INST_0_n_0 ),
        .CO({\add_out[20]_INST_0_n_0 ,\add_out[20]_INST_0_n_1 ,\add_out[20]_INST_0_n_2 ,\add_out[20]_INST_0_n_3 }),
        .CYINIT(1'b0),
        .DI(add_a[23:20]),
        .O(add_out[23:20]),
        .S({\add_out[20]_INST_0_i_1_n_0 ,\add_out[20]_INST_0_i_2_n_0 ,\add_out[20]_INST_0_i_3_n_0 ,\add_out[20]_INST_0_i_4_n_0 }));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[20]_INST_0_i_1 
       (.I0(add_a[23]),
        .I1(add_b[23]),
        .O(\add_out[20]_INST_0_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[20]_INST_0_i_2 
       (.I0(add_a[22]),
        .I1(add_b[22]),
        .O(\add_out[20]_INST_0_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[20]_INST_0_i_3 
       (.I0(add_a[21]),
        .I1(add_b[21]),
        .O(\add_out[20]_INST_0_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[20]_INST_0_i_4 
       (.I0(add_a[20]),
        .I1(add_b[20]),
        .O(\add_out[20]_INST_0_i_4_n_0 ));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \add_out[24]_INST_0 
       (.CI(\add_out[20]_INST_0_n_0 ),
        .CO({\add_out[24]_INST_0_n_0 ,\add_out[24]_INST_0_n_1 ,\add_out[24]_INST_0_n_2 ,\add_out[24]_INST_0_n_3 }),
        .CYINIT(1'b0),
        .DI(add_a[27:24]),
        .O(add_out[27:24]),
        .S({\add_out[24]_INST_0_i_1_n_0 ,\add_out[24]_INST_0_i_2_n_0 ,\add_out[24]_INST_0_i_3_n_0 ,\add_out[24]_INST_0_i_4_n_0 }));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[24]_INST_0_i_1 
       (.I0(add_a[27]),
        .I1(add_b[27]),
        .O(\add_out[24]_INST_0_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[24]_INST_0_i_2 
       (.I0(add_a[26]),
        .I1(add_b[26]),
        .O(\add_out[24]_INST_0_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[24]_INST_0_i_3 
       (.I0(add_a[25]),
        .I1(add_b[25]),
        .O(\add_out[24]_INST_0_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[24]_INST_0_i_4 
       (.I0(add_a[24]),
        .I1(add_b[24]),
        .O(\add_out[24]_INST_0_i_4_n_0 ));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \add_out[28]_INST_0 
       (.CI(\add_out[24]_INST_0_n_0 ),
        .CO({\NLW_add_out[28]_INST_0_CO_UNCONNECTED [3],\add_out[28]_INST_0_n_1 ,\add_out[28]_INST_0_n_2 ,\add_out[28]_INST_0_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,add_a[30:28]}),
        .O(add_out[31:28]),
        .S({\add_out[28]_INST_0_i_1_n_0 ,\add_out[28]_INST_0_i_2_n_0 ,\add_out[28]_INST_0_i_3_n_0 ,\add_out[28]_INST_0_i_4_n_0 }));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[28]_INST_0_i_1 
       (.I0(add_a[31]),
        .I1(add_b[31]),
        .O(\add_out[28]_INST_0_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[28]_INST_0_i_2 
       (.I0(add_a[30]),
        .I1(add_b[30]),
        .O(\add_out[28]_INST_0_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[28]_INST_0_i_3 
       (.I0(add_a[29]),
        .I1(add_b[29]),
        .O(\add_out[28]_INST_0_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[28]_INST_0_i_4 
       (.I0(add_a[28]),
        .I1(add_b[28]),
        .O(\add_out[28]_INST_0_i_4_n_0 ));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \add_out[4]_INST_0 
       (.CI(\add_out[0]_INST_0_n_0 ),
        .CO({\add_out[4]_INST_0_n_0 ,\add_out[4]_INST_0_n_1 ,\add_out[4]_INST_0_n_2 ,\add_out[4]_INST_0_n_3 }),
        .CYINIT(1'b0),
        .DI(add_a[7:4]),
        .O(add_out[7:4]),
        .S({\add_out[4]_INST_0_i_1_n_0 ,\add_out[4]_INST_0_i_2_n_0 ,\add_out[4]_INST_0_i_3_n_0 ,\add_out[4]_INST_0_i_4_n_0 }));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[4]_INST_0_i_1 
       (.I0(add_a[7]),
        .I1(add_b[7]),
        .O(\add_out[4]_INST_0_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[4]_INST_0_i_2 
       (.I0(add_a[6]),
        .I1(add_b[6]),
        .O(\add_out[4]_INST_0_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[4]_INST_0_i_3 
       (.I0(add_a[5]),
        .I1(add_b[5]),
        .O(\add_out[4]_INST_0_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[4]_INST_0_i_4 
       (.I0(add_a[4]),
        .I1(add_b[4]),
        .O(\add_out[4]_INST_0_i_4_n_0 ));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \add_out[8]_INST_0 
       (.CI(\add_out[4]_INST_0_n_0 ),
        .CO({\add_out[8]_INST_0_n_0 ,\add_out[8]_INST_0_n_1 ,\add_out[8]_INST_0_n_2 ,\add_out[8]_INST_0_n_3 }),
        .CYINIT(1'b0),
        .DI(add_a[11:8]),
        .O(add_out[11:8]),
        .S({\add_out[8]_INST_0_i_1_n_0 ,\add_out[8]_INST_0_i_2_n_0 ,\add_out[8]_INST_0_i_3_n_0 ,\add_out[8]_INST_0_i_4_n_0 }));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[8]_INST_0_i_1 
       (.I0(add_a[11]),
        .I1(add_b[11]),
        .O(\add_out[8]_INST_0_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[8]_INST_0_i_2 
       (.I0(add_a[10]),
        .I1(add_b[10]),
        .O(\add_out[8]_INST_0_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[8]_INST_0_i_3 
       (.I0(add_a[9]),
        .I1(add_b[9]),
        .O(\add_out[8]_INST_0_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \add_out[8]_INST_0_i_4 
       (.I0(add_a[8]),
        .I1(add_b[8]),
        .O(\add_out[8]_INST_0_i_4_n_0 ));
endmodule

(* CHECK_LICENSE_TYPE = "risc32_adder_0_0,adder,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "adder,Vivado 2022.2" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (add_a,
    add_b,
    add_out);
  input [31:0]add_a;
  input [31:0]add_b;
  output [31:0]add_out;

  wire [31:0]add_a;
  wire [31:0]add_b;
  wire [31:0]add_out;

  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_adder inst
       (.add_a(add_a),
        .add_b(add_b),
        .add_out(add_out));
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
