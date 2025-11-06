`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao bin
// 
// Create Date: 2024/10/18 13:06:25
// Design Name: 
// Module Name: adder
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


module adder(input  [31:0]      add_a,
             input  [31:0]      add_b,
             output [31:0]     add_out);

assign  add_out =   add_a + add_b;
             
endmodule
