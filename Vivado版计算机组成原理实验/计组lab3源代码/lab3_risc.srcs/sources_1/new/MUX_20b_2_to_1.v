`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao bin
// 
// Create Date: 2024/10/19 11:41:22
// Design Name: 
// Module Name: MUX_20b_2_to_1
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


module MUX_20b_2_to_1(input     [19:0]  data0,
                      input     [19:0]  data1,
                      input             sel  ,          
                      output    [19:0]  result
                      );

assign  result  =   sel ? data1 : data0;

endmodule
