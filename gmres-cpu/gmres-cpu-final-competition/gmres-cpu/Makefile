CXX=g++

CXX_CFLAGS=-O2 -w -std=c++14
CXX_LDFLAGS=

#目录
DIR_OBJ = obj
DIR_SRC = src
DIR_INC = inc

#获取src下所有源文件
CPP_SRCS = $(wildcard $(DIR_SRC)/*.cpp)

#目标文件 $(patsubst 原模式， 目标模式， 文件列表)
AOBJS=$(patsubst $(DIR_SRC)/%.cpp, $(DIR_OBJ)/%.o, $(CPP_SRCS))

#头文件，依赖文件
DEPS=$(wildcard $(DIR_INC)/*.hpp)

# 添加库和头文件路径
LIBS=
INCLUDES= -I./$(DIR_INC)

PROG = gmres

$(DIR_OBJ)/%.o:$(DIR_SRC)/%.cpp
	@mkdir -p $(DIR_OBJ)
	$(CXX) -c $(CXX_CFLAGS) $(INCLUDES)  $< -o $@

$(PROG): $(AOBJS)
	$(CXX) $(CXX_LDFLAGS) $(LIBS) $(INCLUDES)  $^ -o $@

.PHONY:clean
clean:
	rm -f $(DIR_OBJ)/*.o $(PROG)
