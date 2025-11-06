`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao bin
// 
// Create Date: 2024/10/18 13:14:03
// Design Name: 
// Module Name: control_rom
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

module control_rom(  input      [8:0]    inst_ctr,
			         output		         imm_sel,     
			         output              bsel,
			         output              mem_sel,
			         output     [2:0]    alusel,
			         output              w_ena,
			         output              mem_ena);
			         
			         	
	reg [7:0] tem;
	
	assign {imm_sel, bsel, mem_sel, alusel, w_ena, mem_ena} = tem[7:0];
	
	always @(*) begin
		case (inst_ctr)
			12   : tem = 8'b00000010;  	 //add control
//          268  : tem = 8'b00010010;		//sub control		
			4    : tem = 8'b01000010;		//addi control
			260  : tem = 8'b01000010;		//addi control
			64   : tem = 8'b01100010;		//lw control
			72   : tem = 8'b11000001;		//sw control
			default : tem = 0;
		endcase
	end
	
endmodule
