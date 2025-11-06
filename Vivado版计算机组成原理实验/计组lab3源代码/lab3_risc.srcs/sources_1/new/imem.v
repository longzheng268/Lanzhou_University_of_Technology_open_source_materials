`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao Bin
// 
// Create Date: 2024/10/18 12:53:38
// Design Name: 
// Module Name: imem
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module imem(input  	   [31:0] inst_addr,
			output reg [31:0] inst_o);
				
	wire [5:0] addr;
	
	assign addr[5:0] = inst_addr[5:0];
	
	always @(*) begin
		case (addr)
			0  : inst_o = 32'h74300093;   //write 01 to regfile, address equalls 01(addi);
			4  : inst_o = 32'h00200113;
			8  : inst_o = 32'h00300193;
			12 : inst_o = 32'h00400213;
			16 : inst_o = 32'h00500293;
			20 : inst_o = 32'h00600313;
			24 : inst_o = 32'h00700393;
			28 : inst_o = 32'h00800413;
			32 : inst_o = 32'h01008793;    //add 10 to rs01 , write sum to rd15(addi);
			36 : inst_o = 32'h00f08833;	//rs01 + rs15 -> rd16  (add);
			default : inst_o = 0;
		endcase
	end
	
endmodule



