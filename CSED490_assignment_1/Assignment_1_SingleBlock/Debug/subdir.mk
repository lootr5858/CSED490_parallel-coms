################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../reverseArray_singleblock.cu 

OBJS += \
./reverseArray_singleblock.o 

CU_DEPS += \
./reverseArray_singleblock.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -G -g -O0 -gencode arch=compute_75,code=sm_75 -m64 -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_75,code=compute_75 -gencode arch=compute_75,code=sm_75 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


