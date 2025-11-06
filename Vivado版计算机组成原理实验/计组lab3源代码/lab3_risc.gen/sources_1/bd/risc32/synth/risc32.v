//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
//Date        : Wed Nov  5 17:52:47 2025
//Host        : DESKTOP-5VI2UC6 running 64-bit major release  (build 9200)
//Command     : generate_target risc32.bd
//Design      : risc32
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CORE_GENERATION_INFO = "risc32,IP_Integrator,{x_ipVendor=xilinx.com,x_ipLibrary=BlockDiagram,x_ipName=risc32,x_ipVersion=1.00.a,x_ipLanguage=VERILOG,numBlks=19,numReposBlks=19,numNonXlnxBlks=0,numHierBlks=0,maxHierDepth=0,numSysgenBlks=0,numHlsBlks=0,numHdlrefBlks=15,numPkgbdBlks=0,bdsource=USER,synth_mode=OOC_per_IP}" *) (* HW_HANDOFF = "risc32.hwdef" *) 
module risc32
   (SSEG_AN,
    SSEG_CA,
    alu_out_0,
    clk,
    clk_in1_0,
    pc_clr);
  output [7:0]SSEG_AN;
  output [7:0]SSEG_CA;
  output [31:0]alu_out_0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 CLK.CLK CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME CLK.CLK, CLK_DOMAIN risc32_clk, FREQ_HZ 100000000, FREQ_TOLERANCE_HZ 0, INSERT_VIP 0, PHASE 0.0" *) input clk;
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 CLK.CLK_IN1_0 CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME CLK.CLK_IN1_0, CLK_DOMAIN risc32_clk_in1_0, FREQ_HZ 100000000, FREQ_TOLERANCE_HZ 0, INSERT_VIP 0, PHASE 0.0" *) input clk_in1_0;
  input pc_clr;

  wire [19:0]MUX_20b_2_to_1_0_result;
  wire [31:0]MUX_32b_2_to_1_0_result;
  wire [31:0]MUX_32b_2_to_1_1_result;
  wire [4:0]MUX_5b_2_to_1_0_result;
  wire [31:0]adder_0_add_out;
  wire [31:0]alu32_0_alu_out;
  wire [8:0]cancat_control_0_dout;
  wire clk_0_1;
  wire clk_in1_0_1;
  wire clk_wiz_0_clk_out1;
  wire clk_wiz_0_locked;
  wire [31:0]concat_imm_0_dout;
  wire [2:0]control_rom_0_alusel;
  wire control_rom_0_bsel;
  wire control_rom_0_imm_sel;
  wire control_rom_0_mem_ena;
  wire control_rom_0_mem_sel;
  wire control_rom_0_w_ena;
  wire [31:0]dmem_0_d_out;
  wire [31:0]imem_0_inst;
  wire [31:0]pc_0_q;
  wire pc_clr_0_1;
  wire [31:0]regfile_0_reg_data1;
  wire [31:0]regfile_0_reg_data2;
  wire [7:0]segment_0_AN;
  wire [7:0]segment_0_seg_data_o;
  wire [4:0]slice_inst_0_inst_11_7;
  wire [2:0]slice_inst_0_inst_14_12;
  wire [4:0]slice_inst_0_inst_19_15;
  wire [4:0]slice_inst_0_inst_24_20;
  wire slice_inst_0_inst_30;
  wire slice_inst_0_inst_31;
  wire [6:0]slice_inst_0_inst_31_25;
  wire [4:0]slice_inst_0_inst_6_2;
  wire [31:0]xlconstant_0_dout;
  wire [19:0]xlconstant_1_dout;
  wire [19:0]xlconstant_2_dout;

  assign SSEG_AN[7:0] = segment_0_AN;
  assign SSEG_CA[7:0] = segment_0_seg_data_o;
  assign alu_out_0[31:0] = alu32_0_alu_out;
  assign clk_0_1 = clk;
  assign clk_in1_0_1 = clk_in1_0;
  assign pc_clr_0_1 = pc_clr;
  risc32_MUX_20b_2_to_1_0_0 MUX_20b_2_to_1_0
       (.data0(xlconstant_1_dout),
        .data1(xlconstant_2_dout),
        .result(MUX_20b_2_to_1_0_result),
        .sel(slice_inst_0_inst_31));
  risc32_MUX_32b_2_to_1_0_0 MUX_32b_2_to_1_0
       (.data0(regfile_0_reg_data2),
        .data1(concat_imm_0_dout),
        .result(MUX_32b_2_to_1_0_result),
        .sel(control_rom_0_bsel));
  risc32_MUX_32b_2_to_1_1_0 MUX_32b_2_to_1_1
       (.data0(alu32_0_alu_out),
        .data1(dmem_0_d_out),
        .result(MUX_32b_2_to_1_1_result),
        .sel(control_rom_0_mem_sel));
  risc32_MUX_5b_2_to_1_0_1 MUX_5b_2_to_1_0
       (.data0(slice_inst_0_inst_24_20),
        .data1(slice_inst_0_inst_11_7),
        .result(MUX_5b_2_to_1_0_result),
        .sel(control_rom_0_imm_sel));
  risc32_adder_0_0 adder_0
       (.add_a(pc_0_q),
        .add_b(xlconstant_0_dout),
        .add_out(adder_0_add_out));
  risc32_alu32_0_0 alu32_0
       (.alu_a(regfile_0_reg_data1),
        .alu_b(MUX_32b_2_to_1_0_result),
        .alu_out(alu32_0_alu_out),
        .alu_sel(control_rom_0_alusel));
  risc32_cancat_control_0_0 cancat_control_0
       (.dout(cancat_control_0_dout),
        .in0(slice_inst_0_inst_6_2),
        .in1(slice_inst_0_inst_14_12),
        .in2(slice_inst_0_inst_30));
  risc32_clk_wiz_0_0 clk_wiz_0
       (.clk_in1(clk_in1_0_1),
        .clk_out1(clk_wiz_0_clk_out1),
        .locked(clk_wiz_0_locked));
  risc32_concat_imm_0_0 concat_imm_0
       (.dout(concat_imm_0_dout),
        .in0(MUX_5b_2_to_1_0_result),
        .in1(slice_inst_0_inst_31_25),
        .in2(MUX_20b_2_to_1_0_result));
  risc32_control_rom_0_0 control_rom_0
       (.alusel(control_rom_0_alusel),
        .bsel(control_rom_0_bsel),
        .imm_sel(control_rom_0_imm_sel),
        .inst_ctr(cancat_control_0_dout),
        .mem_ena(control_rom_0_mem_ena),
        .mem_sel(control_rom_0_mem_sel),
        .w_ena(control_rom_0_w_ena));
  risc32_dmem_0_0 dmem_0
       (.addr(alu32_0_alu_out),
        .clk(clk_0_1),
        .d_in(regfile_0_reg_data2),
        .d_out(dmem_0_d_out),
        .w_ena(control_rom_0_mem_ena));
  risc32_imem_0_0 imem_0
       (.inst_addr(pc_0_q),
        .inst_o(imem_0_inst));
  risc32_pc_0_0 pc_0
       (.clk(clk_0_1),
        .data_i(adder_0_add_out),
        .pc_clr(pc_clr_0_1),
        .q(pc_0_q));
  risc32_regfile_0_0 regfile_0
       (.clk(clk_0_1),
        .r_addr1(slice_inst_0_inst_19_15),
        .r_addr2(slice_inst_0_inst_24_20),
        .reg_data1(regfile_0_reg_data1),
        .reg_data2(regfile_0_reg_data2),
        .w_addr(slice_inst_0_inst_11_7),
        .w_data(MUX_32b_2_to_1_1_result),
        .w_ena(control_rom_0_w_ena));
  risc32_segment_0_0 segment_0
       (.AN(segment_0_AN),
        .Data_i({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .clk(clk_wiz_0_clk_out1),
        .rst_n(clk_wiz_0_locked),
        .seg_data_o(segment_0_seg_data_o));
  risc32_slice_inst_0_0 slice_inst_0
       (.inst_11_7(slice_inst_0_inst_11_7),
        .inst_14_12(slice_inst_0_inst_14_12),
        .inst_19_15(slice_inst_0_inst_19_15),
        .inst_24_20(slice_inst_0_inst_24_20),
        .inst_30(slice_inst_0_inst_30),
        .inst_31(slice_inst_0_inst_31),
        .inst_31_25(slice_inst_0_inst_31_25),
        .inst_6_2(slice_inst_0_inst_6_2),
        .instruction(imem_0_inst));
  risc32_xlconstant_0_0 xlconstant_0
       (.dout(xlconstant_0_dout));
  risc32_xlconstant_1_0 xlconstant_1
       (.dout(xlconstant_1_dout));
  risc32_xlconstant_2_0 xlconstant_2
       (.dout(xlconstant_2_dout));
endmodule
