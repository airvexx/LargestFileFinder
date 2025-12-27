use anyhow::Context;
use clap::Parser;
use humansize::{format_size, BINARY};
use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "largest-file-finder",
    about = "Find the largest files under a directory"
)]
struct Args {
    /// Root directory to scan (default: /)
    #[arg(default_value = "/")]
    root: PathBuf,

    /// Number of largest files to print
    #[arg(long, default_value_t = 20)]
    top: usize,

    /// Ignore files smaller than this many bytes
    #[arg(long, default_value_t = 0)]
    min_bytes: u64,

    /// Write a pre-built index (JSONL) while scanning
    #[arg(long, value_name = "FILE", conflicts_with = "index_read")]
    index_write: Option<PathBuf>,

    /// Read from an existing index (JSONL) instead of scanning
    #[arg(long, value_name = "FILE", conflicts_with = "index_write")]
    index_read: Option<PathBuf>,

    /// Boolean query expression (AND/OR/NOT, parentheses) over name/path regex + ext/size
    ///
    /// Examples:
    ///   name:/\\.(mp4|mkv)$/ AND size>1GB
    ///   path:/Downloads/ AND NOT name:/\\.part$/
    #[arg(long, value_name = "EXPR")]
    query: Option<String>,

    /// Include filter (case-insensitive regex) applied to full path; repeatable
    #[arg(long, value_name = "REGEX")]
    include: Vec<String>,

    /// Exclude filter (case-insensitive regex) applied to full path; repeatable
    #[arg(long, value_name = "REGEX")]
    exclude: Vec<String>,

    /// Follow symlinks while walking
    #[arg(long, default_value_t = false)]
    follow_symlinks: bool,

    /// Do not cross filesystem boundaries (best-effort)
    #[arg(long, default_value_t = false)]
    one_file_system: bool,

    /// Print progress occasionally (paths skipped / visited)
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct SizedPath {
    size: u64,
    path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    path: String,
    size: u64,
}

#[derive(Debug, Clone)]
struct Matcher {
    query: Option<Expr>,
    include: Vec<Regex>,
    exclude: Vec<Regex>,
}

impl Matcher {
    fn from_args(args: &Args) -> anyhow::Result<Self> {
        let mut include = Vec::with_capacity(args.include.len());
        for pat in &args.include {
            let re = RegexBuilder::new(pat)
                .case_insensitive(true)
                .build()
                .with_context(|| format!("Invalid --include regex: {pat}"))?;
            include.push(re);
        }

        let mut exclude = Vec::with_capacity(args.exclude.len());
        for pat in &args.exclude {
            let re = RegexBuilder::new(pat)
                .case_insensitive(true)
                .build()
                .with_context(|| format!("Invalid --exclude regex: {pat}"))?;
            exclude.push(re);
        }

        let query = match args.query.as_deref() {
            Some(expr_str) => Some(ParserExpr::new(expr_str).parse().with_context(|| {
                format!("Invalid --query expression: {expr_str}")
            })?),
            None => None,
        };

        Ok(Self {
            query,
            include,
            exclude,
        })
    }

    fn matches_path_str(&self, path: &str, size: u64) -> bool {
        for re in &self.exclude {
            if re.is_match(path) {
                return false;
            }
        }

        // If any --include filters are provided, the path must match at least one.
        if !self.include.is_empty() && !self.include.iter().any(|re| re.is_match(path)) {
            return false;
        }

        if let Some(expr) = &self.query {
            return expr.eval(path, size);
        }

        true
    }
}

impl Ord for SizedPath {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural (max-heap) ordering: larger sizes first.
        // Tie-breaker ensures deterministic ordering.
        self.size
            .cmp(&other.size)
            // Reverse path ordering so that, for equal sizes, lexicographically *smaller* paths win
            // when we maintain a min-heap via Reverse<SizedPath>.
            .then_with(|| other.path.cmp(&self.path))
    }
}

impl PartialOrd for SizedPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let matcher = Matcher::from_args(&args)?;

    let root = args.root;

    let top_n = args.top.max(1);
    let min_bytes = args.min_bytes;

    if args.index_read.is_none() && !root.exists() {
        anyhow::bail!("Root path does not exist: {}", root.display());
    }

    // If the user points directly at a file, treat it as a 1-item scan.
    if args.index_read.is_none() && root.is_file() {
        let md = std::fs::metadata(&root)?;
        let size = md.len();
        let path_str = root.to_string_lossy();
        if size >= min_bytes && matcher.matches_path_str(&path_str, size) {
            println!("#1\t{}\t{}", format_size(size, BINARY), root.display());
        } else {
            println!(
                "No matching files >= {} bytes found at {}",
                min_bytes,
                root.display()
            );
        }
        return Ok(());
    }

    let mut top_files: BinaryHeap<Reverse<SizedPath>> = BinaryHeap::with_capacity(top_n);

    if let Some(index_path) = &args.index_read {
        read_index_and_collect(
            index_path,
            &matcher,
            top_n,
            min_bytes,
            &mut top_files,
            args.verbose,
        )?;
    } else {
        let mut index_writer = match &args.index_write {
            Some(path) => Some(BufWriter::new(
                File::create(path).with_context(|| format!("Failed to create index file: {}", path.display()))?,
            )),
            None => None,
        };

        scan_filesystem_and_collect(
            &root,
            args.follow_symlinks,
            args.one_file_system,
            args.verbose,
            &matcher,
            top_n,
            min_bytes,
            &mut top_files,
            index_writer.as_mut(),
        )?;

        if let Some(mut w) = index_writer {
            w.flush().context("Failed to flush index file")?;
        }
    }

    let mut results: Vec<SizedPath> = top_files
        .into_iter()
        .map(|Reverse(sp)| sp)
        .collect();

    results.sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.path.cmp(&b.path)));

    if results.is_empty() {
        if let Some(index_path) = &args.index_read {
            println!("No matching files found in index {}", index_path.display());
        } else {
            println!("No matching files found under {}", root.display());
        }
        return Ok(());
    }

    for (idx, item) in results.iter().enumerate() {
        println!(
            "#{}\t{}\t{}",
            idx + 1,
            format_size(item.size, BINARY),
            item.path.display()
        );
    }

    Ok(())
}

fn scan_filesystem_and_collect(
    root: &Path,
    follow_symlinks: bool,
    one_file_system: bool,
    verbose: bool,
    matcher: &Matcher,
    top_n: usize,
    min_bytes: u64,
    top_files: &mut BinaryHeap<Reverse<SizedPath>>,
    mut index_writer: Option<&mut BufWriter<File>>,
) -> anyhow::Result<()> {
    let root_dev = if one_file_system {
        device_id(root).ok()
    } else {
        None
    };

    let mut visited: u64 = 0;
    let mut skipped: u64 = 0;

    let walker = WalkDir::new(root)
        .follow_links(follow_symlinks)
        .same_file_system(one_file_system)
        .into_iter();

    for entry_result in walker {
        let entry = match entry_result {
            Ok(e) => e,
            Err(_err) => {
                skipped += 1;
                continue;
            }
        };

        let md = match entry.metadata() {
            Ok(m) => m,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        if !md.is_file() {
            continue;
        }

        if let (Some(root_dev), true) = (root_dev.as_ref(), one_file_system) {
            if let Ok(dev) = file_device_id(entry.path()) {
                if &dev != root_dev {
                    continue;
                }
            }
        }

        visited += 1;
        let size = md.len();
        let path_cow = entry.path().to_string_lossy();

        if let Some(w) = index_writer.as_mut() {
            let rec = IndexEntry {
                path: path_cow.to_string(),
                size,
            };
            serde_json::to_writer(w.by_ref(), &rec)
                .context("Failed to write JSON record to index")?;
            w.write_all(b"\n")
                .context("Failed to write newline to index")?;
        }

        if size < min_bytes {
            continue;
        }

        if !matcher.matches_path_str(path_cow.as_ref(), size) {
            continue;
        }

        let candidate = SizedPath {
            size,
            path: entry.path().to_path_buf(),
        };

        consider_candidate(top_files, top_n, candidate);

        if verbose && visited % 200_000 == 0 {
            let current_floor = top_files
                .peek()
                .map(|Reverse(sp)| sp.size)
                .unwrap_or(0);
            eprintln!(
                "Visited: {visited}, skipped: {skipped}, collected: {}, current top-floor: {} ({current_floor} bytes)",
                top_files.len(),
                format_size(current_floor, BINARY)
            );
        }
    }

    Ok(())
}

fn read_index_and_collect(
    index_path: &Path,
    matcher: &Matcher,
    top_n: usize,
    min_bytes: u64,
    top_files: &mut BinaryHeap<Reverse<SizedPath>>,
    verbose: bool,
) -> anyhow::Result<()> {
    let file = File::open(index_path)
        .with_context(|| format!("Failed to open index file: {}", index_path.display()))?;
    let reader = BufReader::new(file);

    let mut read_lines: u64 = 0;
    let mut parsed: u64 = 0;
    let mut skipped: u64 = 0;

    for line_res in reader.lines() {
        read_lines += 1;
        let line = match line_res {
            Ok(l) => l,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let rec: IndexEntry = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        parsed += 1;

        if rec.size < min_bytes {
            continue;
        }

        if !matcher.matches_path_str(&rec.path, rec.size) {
            continue;
        }

        let candidate = SizedPath {
            size: rec.size,
            path: PathBuf::from(rec.path),
        };
        consider_candidate(top_files, top_n, candidate);

        if verbose && parsed % 500_000 == 0 {
            let current_floor = top_files
                .peek()
                .map(|Reverse(sp)| sp.size)
                .unwrap_or(0);
            eprintln!(
                "Index lines: {read_lines}, parsed: {parsed}, skipped: {skipped}, collected: {}, current top-floor: {} ({current_floor} bytes)",
                top_files.len(),
                format_size(current_floor, BINARY)
            );
        }
    }

    Ok(())
}

fn consider_candidate(
    top_files: &mut BinaryHeap<Reverse<SizedPath>>,
    top_n: usize,
    candidate: SizedPath,
) {
    if top_files.len() < top_n {
        top_files.push(Reverse(candidate));
    } else if let Some(Reverse(current_smallest)) = top_files.peek() {
        if candidate.cmp(current_smallest) == Ordering::Greater {
            top_files.pop();
            top_files.push(Reverse(candidate));
        }
    }
}

#[derive(Debug, Clone)]
enum Expr {
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    Pred(Predicate),
}

impl Expr {
    fn eval(&self, path: &str, size: u64) -> bool {
        match self {
            Expr::And(a, b) => a.eval(path, size) && b.eval(path, size),
            Expr::Or(a, b) => a.eval(path, size) || b.eval(path, size),
            Expr::Not(inner) => !inner.eval(path, size),
            Expr::Pred(p) => p.eval(path, size),
        }
    }
}

#[derive(Debug, Clone)]
enum CmpOp {
    Lt,
    Lte,
    Gt,
    Gte,
    Eq,
}

#[derive(Debug, Clone)]
enum Predicate {
    PathRegex(Regex),
    NameRegex(Regex),
    ExtEq(String),
    SizeCmp { op: CmpOp, bytes: u64 },
}

impl Predicate {
    fn eval(&self, path: &str, size: u64) -> bool {
        match self {
            Predicate::PathRegex(re) => re.is_match(path),
            Predicate::NameRegex(re) => Path::new(path)
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|name| re.is_match(name)),
            Predicate::ExtEq(ext) => Path::new(path)
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case(ext)),
            Predicate::SizeCmp { op, bytes } => match op {
                CmpOp::Lt => size < *bytes,
                CmpOp::Lte => size <= *bytes,
                CmpOp::Gt => size > *bytes,
                CmpOp::Gte => size >= *bytes,
                CmpOp::Eq => size == *bytes,
            },
        }
    }
}

struct ParserExpr<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> ParserExpr<'a> {
    fn new(s: &'a str) -> Self {
        Self { s, i: 0 }
    }

    fn parse(mut self) -> anyhow::Result<Expr> {
        let expr = self.parse_or()?;
        self.skip_ws();
        if self.i != self.s.len() {
            anyhow::bail!("Unexpected trailing input at byte {}", self.i);
        }
        Ok(expr)
    }

    fn parse_or(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_and()?;
        loop {
            self.skip_ws();
            if self.consume_op("||") || self.consume_kw("OR") {
                let right = self.parse_and()?;
                left = Expr::Or(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> anyhow::Result<Expr> {
        let mut left = self.parse_unary()?;
        loop {
            self.skip_ws();
            if self.consume_op("&&") || self.consume_kw("AND") {
                let right = self.parse_unary()?;
                left = Expr::And(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> anyhow::Result<Expr> {
        self.skip_ws();
        if self.consume_op("!") || self.consume_kw("NOT") {
            let inner = self.parse_unary()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> anyhow::Result<Expr> {
        self.skip_ws();
        if self.consume_op("(") {
            let inner = self.parse_or()?;
            self.skip_ws();
            if !self.consume_op(")") {
                anyhow::bail!("Expected ')' at byte {}", self.i);
            }
            return Ok(inner);
        }

        let pred = self.parse_predicate()?;
        Ok(Expr::Pred(pred))
    }

    fn parse_predicate(&mut self) -> anyhow::Result<Predicate> {
        self.skip_ws();

        if self.peek_char() == Some('/') {
            let re = self.parse_regex_literal()?;
            return Ok(Predicate::PathRegex(re));
        }

        if self.consume_kw("name") {
            self.expect_char(':')?;
            let re = self.parse_regex_literal()?;
            return Ok(Predicate::NameRegex(re));
        }

        if self.consume_kw("path") {
            self.expect_char(':')?;
            let re = self.parse_regex_literal()?;
            return Ok(Predicate::PathRegex(re));
        }

        if self.consume_kw("ext") {
            self.expect_char(':')?;
            let val = self.parse_bare_value()?;
            if val.is_empty() {
                anyhow::bail!("ext: requires a value");
            }
            return Ok(Predicate::ExtEq(val));
        }

        if self.consume_kw("size") {
            let op = self.parse_cmp_op()?;
            let val = self.parse_bare_value()?;
            let bytes = parse_size_bytes(&val).with_context(|| format!("Invalid size literal: {val}"))?;
            return Ok(Predicate::SizeCmp { op, bytes });
        }

        anyhow::bail!("Expected predicate at byte {}", self.i)
    }

    fn parse_cmp_op(&mut self) -> anyhow::Result<CmpOp> {
        self.skip_ws();
        if self.consume_op(">=") {
            return Ok(CmpOp::Gte);
        }
        if self.consume_op("<=") {
            return Ok(CmpOp::Lte);
        }
        if self.consume_op(">") {
            return Ok(CmpOp::Gt);
        }
        if self.consume_op("<") {
            return Ok(CmpOp::Lt);
        }
        if self.consume_op("=") {
            return Ok(CmpOp::Eq);
        }
        anyhow::bail!("Expected comparison operator after size at byte {}", self.i)
    }

    fn parse_regex_literal(&mut self) -> anyhow::Result<Regex> {
        self.skip_ws();
        if self.peek_char() != Some('/') {
            anyhow::bail!("Expected regex literal starting with '/' at byte {}", self.i);
        }
        self.i += 1; // skip '/'

        let mut pat = String::new();
        // We treat '/' as the delimiter; a '/' can be included by escaping it as '\/'.
        // All other backslashes must be preserved (e.g. /\./, /\d+/) so the regex semantics stay intact.
        while self.i < self.s.len() {
            let rest = &self.s[self.i..];
            let Some(c) = rest.chars().next() else {
                break;
            };

            if c == '/' {
                self.i += 1;
                let re = Regex::new(&pat).with_context(|| format!("Invalid regex: /{pat}/"))?;
                return Ok(re);
            }

            if c == '\\' {
                // Preserve backslashes, but allow escaping the delimiter.
                self.i += 1;
                if self.i >= self.s.len() {
                    anyhow::bail!("Unterminated escape in regex literal");
                }
                let rest2 = &self.s[self.i..];
                let next = rest2.chars().next().unwrap();
                self.i += next.len_utf8();
                if next == '/' {
                    pat.push('/');
                } else {
                    pat.push('\\');
                    pat.push(next);
                }
                continue;
            }

            self.i += c.len_utf8();
            pat.push(c);
        }

        anyhow::bail!("Unterminated regex literal")
    }

    fn parse_bare_value(&mut self) -> anyhow::Result<String> {
        self.skip_ws();
        let start = self.i;
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() || c == ')' || c == '(' {
                break;
            }
            self.i += c.len_utf8();
        }
        Ok(self.s[start..self.i].to_string())
    }

    fn expect_char(&mut self, expected: char) -> anyhow::Result<()> {
        self.skip_ws();
        if self.peek_char() == Some(expected) {
            self.i += expected.len_utf8();
            Ok(())
        } else {
            anyhow::bail!("Expected '{expected}' at byte {}", self.i)
        }
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.i += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.s[self.i..].chars().next()
    }

    fn consume_op(&mut self, op: &str) -> bool {
        if self.s[self.i..].starts_with(op) {
            self.i += op.len();
            true
        } else {
            false
        }
    }

    fn consume_kw(&mut self, kw: &str) -> bool {
        self.skip_ws();
        let rest = &self.s[self.i..];
        let mut chars = rest.chars();
        let mut taken = String::new();
        while let Some(c) = chars.next() {
            if c.is_alphanumeric() || c == '_' {
                taken.push(c);
            } else {
                break;
            }
        }
        if taken.eq_ignore_ascii_case(kw) {
            self.i += taken.len();
            true
        } else {
            false
        }
    }
}

fn parse_size_bytes(s: &str) -> anyhow::Result<u64> {
    let t = s.trim();
    if t.is_empty() {
        anyhow::bail!("empty size");
    }

    let mut digits_end = 0;
    for (idx, ch) in t.char_indices() {
        if ch.is_ascii_digit() {
            digits_end = idx + 1;
        } else {
            break;
        }
    }

    if digits_end == 0 {
        anyhow::bail!("size must start with digits");
    }

    let num: u64 = t[..digits_end].parse()?;
    let unit = t[digits_end..].trim().to_ascii_lowercase();
    let mul: u64 = match unit.as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024_u64.pow(2),
        "g" | "gb" | "gib" => 1024_u64.pow(3),
        "t" | "tb" | "tib" => 1024_u64.pow(4),
        _ => anyhow::bail!("unknown size unit: {unit}"),
    };
    num.checked_mul(mul)
        .context("size overflow")
}

#[cfg(unix)]
fn device_id(path: &Path) -> std::io::Result<u64> {
    use std::os::unix::fs::MetadataExt;
    let md = std::fs::metadata(path)?;
    Ok(md.dev())
}

#[cfg(unix)]
fn file_device_id(path: &Path) -> std::io::Result<u64> {
    use std::os::unix::fs::MetadataExt;
    let md = std::fs::metadata(path)?;
    Ok(md.dev())
}

#[cfg(not(unix))]
fn device_id(_path: &Path) -> std::io::Result<u64> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "device id unsupported on this platform",
    ))
}

#[cfg(not(unix))]
fn file_device_id(_path: &Path) -> std::io::Result<u64> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "device id unsupported on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_literal_preserves_backslashes() {
        let expr = ParserExpr::new(r"name:/\.(mp4|mkv)$/").parse().unwrap();
        // Should match a literal dot.
        assert!(expr.eval("/tmp/movie.mp4", 0));
        // Should not match if the extension is preceded by another character (would match if the '\\.' got stripped).
        assert!(!expr.eval("/tmp/movieXmp4", 0));
    }

    #[test]
    fn regex_literal_can_escape_delimiter_slash() {
        let expr = ParserExpr::new(r"path:/foo\/bar/").parse().unwrap();
        assert!(expr.eval("/tmp/foo/bar/baz", 0));
        assert!(!expr.eval("/tmp/fooXbar/baz", 0));
    }

    #[test]
    fn include_filters_are_any_match() {
        let matcher = Matcher {
            query: None,
            include: vec![Regex::new("foo").unwrap(), Regex::new("bar").unwrap()],
            exclude: vec![],
        };
        assert!(matcher.matches_path_str("/tmp/foo.txt", 0));
        assert!(matcher.matches_path_str("/tmp/bar.txt", 0));
        assert!(!matcher.matches_path_str("/tmp/baz.txt", 0));
    }

    #[test]
    fn equal_size_ties_keep_lexicographically_smallest_path() {
        let mut heap: BinaryHeap<Reverse<SizedPath>> = BinaryHeap::with_capacity(1);

        consider_candidate(
            &mut heap,
            1,
            SizedPath {
                size: 10,
                path: PathBuf::from("b"),
            },
        );
        consider_candidate(
            &mut heap,
            1,
            SizedPath {
                size: 10,
                path: PathBuf::from("a"),
            },
        );

        let kept = heap.pop().unwrap().0;
        assert_eq!(kept.path, PathBuf::from("a"));
    }
}
