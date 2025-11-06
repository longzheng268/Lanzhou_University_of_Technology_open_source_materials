vlib work
vlib riviera

vlib riviera/xpm
vlib riviera/xil_defaultlib
vlib riviera/xlconstant_v1_1_7

vmap xpm riviera/xpm
vmap xil_defaultlib riviera/xil_defaultlib
vmap xlconstant_v1_1_7 riviera/xlconstant_v1_1_7

vlog -work xpm  -sv2k12 "+incdir+../../../../lab3_risc.gen/sources_1/bd/risc32/ipshared/7698" \
"C:/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv" \

vcom -work xpm -93  \
"C:/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_VCOMP.vhd" \

vlog -work xil_defaultlib  -v2k5 "+incdir+../../../../lab3_risc.gen/sources_1/bd/risc32/ipshared/7698" \
"../../../bd/risc32/ip/risc32_adder_0_0/sim/risc32_adder_0_0.v" \

vlog -work xlconstant_v1_1_7  -v2k5 "+incdir+../../../../lab3_risc.gen/sources_1/bd/risc32/ipshared/7698" \
"../../../../lab3_risc.gen/sources_1/bd/risc32/ipshared/badb/hdl/xlconstant_v1_1_vl_rfs.v" \

vlog -work xil_defaultlib  -v2k5 "+incdir+../../../../lab3_risc.gen/sources_1/bd/risc32/ipshared/7698" \
"../../../bd/risc32/ip/risc32_xlconstant_0_0/sim/risc32_xlconstant_0_0.v" \
"../../../bd/risc32/ip/risc32_slice_inst_0_0/sim/risc32_slice_inst_0_0.v" \
"../../../bd/risc32/ip/risc32_cancat_control_0_0/sim/risc32_cancat_control_0_0.v" \
"../../../bd/risc32/ip/risc32_control_rom_0_0/sim/risc32_control_rom_0_0.v" \
"../../../bd/risc32/ip/risc32_imem_0_0/sim/risc32_imem_0_0.v" \
"../../../bd/risc32/ip/risc32_segment_0_0/sim/risc32_segment_0_0.v" \
"../../../bd/risc32/ip/risc32_clk_wiz_0_0/risc32_clk_wiz_0_0_clk_wiz.v" \
"../../../bd/risc32/ip/risc32_clk_wiz_0_0/risc32_clk_wiz_0_0.v" \
"../../../bd/risc32/ip/risc32_pc_0_0/sim/risc32_pc_0_0.v" \
"../../../bd/risc32/ip/risc32_alu32_0_0/sim/risc32_alu32_0_0.v" \
"../../../bd/risc32/ip/risc32_regfile_0_0/sim/risc32_regfile_0_0.v" \
"../../../bd/risc32/ip/risc32_concat_imm_0_0/sim/risc32_concat_imm_0_0.v" \
"../../../bd/risc32/ip/risc32_MUX_20b_2_to_1_0_0/sim/risc32_MUX_20b_2_to_1_0_0.v" \
"../../../bd/risc32/ip/risc32_xlconstant_1_0/sim/risc32_xlconstant_1_0.v" \
"../../../bd/risc32/ip/risc32_xlconstant_2_0/sim/risc32_xlconstant_2_0.v" \
"../../../bd/risc32/ip/risc32_MUX_32b_2_to_1_0_0/sim/risc32_MUX_32b_2_to_1_0_0.v" \
"../../../bd/risc32/ip/risc32_dmem_0_0/sim/risc32_dmem_0_0.v" \
"../../../bd/risc32/ip/risc32_MUX_32b_2_to_1_1_0/sim/risc32_MUX_32b_2_to_1_1_0.v" \
"../../../bd/risc32/ip/risc32_MUX_5b_2_to_1_0_1/sim/risc32_MUX_5b_2_to_1_0_1.v" \
"../../../bd/risc32/sim/risc32.v" \

vlog -work xil_defaultlib \
"glbl.v"

