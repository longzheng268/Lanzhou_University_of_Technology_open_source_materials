//
module alu32(input [31:0] alu_a, alu_b,
			 input		alu_sel,
			 output	[31:0] alu_out);
			 
	assign alu_out = alu_sel ? (alu_a - alu_b) : (alu_a + alu_b);
	
endmodule
