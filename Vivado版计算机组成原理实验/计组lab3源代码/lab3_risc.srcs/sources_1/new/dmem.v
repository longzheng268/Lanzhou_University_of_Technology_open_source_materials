`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut    
// Engineer: Yao bin
// 
// Create Date: 2024/10/18 13:28:29
// Design Name: 
// Module Name: dmem
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: dmem深度128，存储空间有限
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module dmem(input clk, w_ena,
			input [31:0] addr, d_in,
			output [31:0] d_out);
			
	reg [31:0] mem [128:0];
	
	assign d_out = mem[addr];
	
	always @(posedge clk)
		if (w_ena) mem[addr] <= d_in;
		
endmodule
