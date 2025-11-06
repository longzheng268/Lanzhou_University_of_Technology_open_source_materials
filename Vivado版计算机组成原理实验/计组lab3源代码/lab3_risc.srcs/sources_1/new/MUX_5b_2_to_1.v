`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao bin
// 
// Create Date: 2024/10/19 11:50:23
// Design Name: 
// Module Name: MUX_5b_2_to_1
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


module MUX_5b_2_to_1(input     [4:0]  data0,
                     input     [4:0]  data1,
                     input            sel  ,          
                     output    [4:0]  result
                      );

assign  result  =   sel ? data1 : data0;

endmodule
