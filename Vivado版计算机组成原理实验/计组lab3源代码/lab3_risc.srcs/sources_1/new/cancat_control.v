`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao bin
// 
// Create Date: 2024/10/21 16:17:50
// Design Name: 
// Module Name: cancat_control
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


module cancat_control(  input       [4:0]   in0     ,
                        input       [2:0]   in1     ,
                        input               in2     , 
                        output      [8:0]   dout    );

assign  dout    =   {in2,in1,in0};

endmodule
