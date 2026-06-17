import re
import os

# 1. 待审查的 AI 高频敏感词列表
SENSITIVE_WORDS = [
    "delve", "tapestry", "landscape", "pivotal", "crucial", 
    "foster", "showcase", "testament", "navigate", "leverage", 
    "realm", "embark", "underscore", "multifaceted", "nuanced", 
    "comprehensive", "robust", "intricate", "cornerstone", "paradigm", 
    "synergy", "holistic", "streamline", "cutting-edge", "groundbreaking"
]

# 2. 废话开场白 (Throat-clearing phrases)
THROAT_CLEARING = [
    r"\bin the realm of\b",
    r"\bit is important to note that\b",
    r"\bit is worth mentioning that\b",
    r"\bin today's rapidly evolving\b",
    r"\bserves as a testament to\b",
    r"\bit goes without saying that\b",
    r"\bin order to\b",
    r"\bit should be noted that\b",
    r"\bas a matter of fact\b",
    r"\bwhen it comes to\b",
    r"\bat the end of the day\b",
    r"\bwith that being said\b"
]

def analyze_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return False
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    print("==========================================================")
    print(f"正在审计学术论文文本质量: {filepath}")
    print("==========================================================")
    
    passed = True
    
    # A. 敏感词扫描 (生态学 landscape 和 统计学 robust 在专业语境下除外)
    print("\n[A] 敏感词扫描:")
    found_words = {}
    for word in SENSITIVE_WORDS:
        # 使用正则表达式寻找独立单词
        pattern = re.compile(rf"\b{word}s?\b", re.IGNORECASE)
        matches = pattern.findall(content)
        if matches:
            # 豁免规则判断
            if word == "landscape" or word == "robust":
                print(f"  - 发现专业敏感词 '{word}' 共 {len(matches)} 次 (生态/统计专业领域可酌情豁免，请确认)")
            else:
                found_words[word] = len(matches)
                passed = False
                
    if found_words:
        for w, count in found_words.items():
            print(f"  [FAIL] 发现高频禁用词 '{w}': {count} 次")
    else:
        print("  [PASS] 敏感词自检通过 (无禁用词)！")
        
    # B. 废话开场白扫描
    print("\n[B] 过渡废话/开场白扫描:")
    found_tc = 0
    for phrase in THROAT_CLEARING:
        pattern = re.compile(phrase, re.IGNORECASE)
        matches = pattern.findall(content)
        if matches:
            print(f"  [FAIL] 发现废话开场白/过渡词: '{phrase}' {len(matches)} 次")
            found_tc += len(matches)
            passed = False
    if found_tc == 0:
        print("  [PASS] 废话开场白自检通过！")
        
    # C. 标点符号扫描 (破折号与分号)
    print("\n[C] 标点符号与格式检查:")
    # 排除 Markdown 表格分隔线 (如 |---|---| ) 与水平分割线 (如 ---) 导致的误判
    cleaned_content = re.sub(r'^\s*\|(?:\s*:?-+:?\s*\|)+\s*$', '', content, flags=re.MULTILINE)
    cleaned_content = re.sub(r'^\s*---+\s*$', '', cleaned_content, flags=re.MULTILINE)
    em_dashes = len(re.findall(r"—|--", cleaned_content))
    semicolons = content.count(';')
    
    print(f"  - 破折号 (em dash) 数量: {em_dashes} 个 (限额: ≤ 3)")
    print(f"  - 分号 (semicolon) 数量: {semicolons} 个 (建议: 每1000字内 ≤ 2个)")
    
    if em_dashes > 3:
        print("  [FAIL] 破折号数量超额！请用逗号或括号代替。")
        passed = False
    else:
        print("  [PASS] 破折号数量符合要求。")
        
    # D. 段落与句式长短变异性 (Burstiness)
    print("\n[D] 句长变异性分析 (Sentence Length Burstiness):")
    # 粗略分割句子
    sentences = re.split(r'\. |\? |\! ', content)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
    
    if len(sentences) > 5:
        lengths = [len(s.split()) for s in sentences]
        # 计算滑动窗口，判断是否有连续 5 个句子句长过窄
        narrow_count = 0
        for i in range(len(lengths) - 4):
            window = lengths[i:i+5]
            if max(window) - min(window) <= 4:
                narrow_count += 1
                
        if narrow_count > 0:
            print(f"  [WARN] 警告: 发现 {narrow_count} 处连续 5 句句长极度均一 (缺少句式变化，易显单调)")
        else:
            print("  [PASS] 句长变异性通过，长短句结合自然。")
    else:
        print("  - 句子样本量较少，跳过句长变异性分析。")
        
    print("\n==========================================================")
    if passed:
        print("  [SUCCESS] 恭喜！学术写作合规审计通过！")
    else:
        print("  [FAIL] 存在不合规项，请根据上述提示修正文本。")
    print("==========================================================")
    return passed

if __name__ == "__main__":
    import sys
    target = "../manuscript/03_results.md"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    analyze_file(target)
