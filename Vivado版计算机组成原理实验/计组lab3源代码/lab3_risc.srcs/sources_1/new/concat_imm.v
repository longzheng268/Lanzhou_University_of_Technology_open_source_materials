`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao bin
// 
// Create Date: 2024/10/21 16:23:47
// Design Name: 
// Module Name: concat_imm
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


module concat_imm(  input       [4:0]   in0     ,
                    input       [6:0]   in1     ,
                    input       [19:0]  in2     ,
                    output      [31:0]  dout    );

assign  dout    =   {in2,in1,in0};

endmodule
