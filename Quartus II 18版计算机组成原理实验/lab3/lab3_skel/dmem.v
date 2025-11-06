//
module dmem(input clk, w_ena,
			input [31:0] addr, d_in,
			output [31:0] d_out);
			
	reg [31:0] mem [15:0];
	
	assign d_out = mem[addr];
	
	always @(posedge clk)
		if (w_ena) mem[addr] <= d_in;
		
endmodule

			