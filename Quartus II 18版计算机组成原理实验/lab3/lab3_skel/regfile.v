//
module  regfile(input		  clk,
				input		  w_ena,
				input  [4:0]  r_addr1, r_addr2, w_addr,
				input  [31:0] w_data,
				output [31:0] reg_data1, reg_data2);
					
	reg [31:0] reg_array [31:0];
	
	assign reg_data1 = (r_addr1 != 0) ? reg_array[r_addr1] : 0;
	assign reg_data2 = (r_addr2 != 0) ? reg_array[r_addr2] : 0;
	
	always @(posedge clk) 
		if (w_ena) reg_array[w_addr] <= w_data;
		
endmodule
					
					