import sys
import argparse
import numpy

# 浮点数比较的相对和绝对容忍度
# 可以根据赛题要求调整
RTOL = 1e-5
ATOL = 1e-6

def is_close(a, b, abs_tol=ATOL, rel_tol=RTOL):
    # 等价于 Python math.isclose 判定
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))

def compare_files(file1, file2):
    """
    Compares two files line by line, with special handling for floating-point numbers.
    """
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            for line_num, (line1, line2) in enumerate(zip(f1, f2), 1):
                tokens1 = line1.strip().split()
                tokens2 = line2.strip().split()

                if len(tokens1) != len(tokens2):
                    print(f"FAIL: Line {line_num} has different number of elements.")
                    return False

                for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
                    try:
                        # 尝试将 token 转换为浮点数
                        v1 = float(t1)
                        v2 = float(t2)
                        # 使用 numpy.isclose 的逻辑进行比较
                        if not is_close(v1,v2):
                            print(f"FAIL: Mismatch at Line {line_num}, Element {i+1}.")
                            print(f"      Expected: {t2}")
                            print(f"      Got:      {t1}")
                            return False
                    except ValueError:
                        # 如果不是数字，则直接进行字符串比较
                        if t1 != t2:
                            print(f"FAIL: Mismatch at Line {line_num}, Element {i+1}.")
                            print(f"      Expected: {t2}")
                            print(f"      Got:      {t1}")
                            return False
            
            # 检查文件行数是否一致
            if sum(1 for _ in f1) != sum(1 for _ in f2):
                print("FAIL: Files have a different number of lines.")
                return False

    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found.")
        return False
    
    print("PASS: The output is correct.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two output files for correctness.")
    parser.add_argument("student_output", help="Path to the student's output file.")
    parser.add_argument("reference_output", help="Path to the reference (golden) output file.")
    
    args = parser.parse_args()

    if not compare_files(args.student_output, args.reference_output):
        sys.exit(1) # 以非零状态码退出，方便脚本判断
    sys.exit(0)
