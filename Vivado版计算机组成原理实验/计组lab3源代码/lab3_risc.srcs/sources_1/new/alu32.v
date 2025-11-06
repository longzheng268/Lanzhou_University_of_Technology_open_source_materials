`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao bin
// 
// Create Date: 2024/10/18 18:11:36
// Design Name: 
// Module Name: alu32
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


module alu32(input          [31:0]  alu_a   ,
             input          [31:0]  alu_b   ,
             input          [2:0]   alu_sel ,
             output     reg [31:0]  alu_out 
              );

always @(*)
    begin
        case(alu_sel)
            3'd0    :   alu_out =   alu_a + alu_b;
            3'd1    :   alu_out =   alu_a & alu_b;
            3'd2    :   alu_out =   alu_a << alu_b[4:0];
            3'd3    :   alu_out =   $signed(alu_a) >>> alu_b[4:0];
            3'd4    :   alu_out =   $signed(alu_a) < $signed(alu_b); 
            3'd5    :   alu_out =   alu_a ^ alu_b;
            default :   alu_out =   32'd0;            
        endcase
    end
endmodule
