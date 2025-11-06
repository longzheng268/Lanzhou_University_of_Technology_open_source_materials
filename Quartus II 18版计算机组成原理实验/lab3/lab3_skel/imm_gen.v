//
module imm_gen(input [11:0] imm_fld,
			   output [31:0] imm);

	assign imm = {{20{imm_fld[11]}},imm_fld};
	
endmodule
