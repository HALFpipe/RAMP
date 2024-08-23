from pathlib import Path

import pandas as pd

from gwas.compression.pipe import CompressedTextWriter
from gwas.meta.rsid_matching import match_ids

dbsnp_file_contents = """##fileformat=VCFv4.0
##FILTER=<ID=PASS,Description="All filters passed">
##fileDate=20141009
##source=dbSNP
##dbSNP_BUILD_ID=142
##reference=GRCh37.p13
##phasing=partial
##variationPropertyDocumentationUrl=ftp://ftp.ncbi.nlm.nih.gov/snp/specs/dbSNP_BitField_latest.pdf\t
##INFO=<ID=RS,Number=1,Type=Integer,Description="dbSNP ID (i.e. rs number)">
##INFO=<ID=RSPOS,Number=1,Type=Integer,Description="Chr position reported in dbSNP">
##INFO=<ID=RV,Number=0,Type=Flag,Description="RS orientation is reversed">
##INFO=<ID=VP,Number=1,Type=String,Description="Variation Property.  Documentation is at ftp://ftp.ncbi.nlm.nih.gov/snp/specs/dbSNP_BitField_latest.pdf">
##INFO=<ID=GENEINFO,Number=1,Type=String,Description="Pairs each of gene symbol:gene id.  The gene symbol and id are delimited by a colon (:) and each pair is delimited by a vertical bar (|)">
##INFO=<ID=dbSNPBuildID,Number=1,Type=Integer,Description="First dbSNP Build for RS">
##INFO=<ID=SAO,Number=1,Type=Integer,Description="Variant Allele Origin: 0 - unspecified, 1 - Germline, 2 - Somatic, 3 - Both">
##INFO=<ID=SSR,Number=1,Type=Integer,Description="Variant Suspect Reason Codes (may be more than one value added together) 0 - unspecified, 1 - Paralog, 2 - byEST, 4 - oldAlign, 8 - Para_EST, 16 - 1kg_failed, 1024 - other">
##INFO=<ID=WGT,Number=1,Type=Integer,Description="Weight, 00 - unmapped, 1 - weight 1, 2 - weight 2, 3 - weight 3 or more">
##INFO=<ID=VC,Number=1,Type=String,Description="Variation Class">
##INFO=<ID=PM,Number=0,Type=Flag,Description="Variant is Precious(Clinical,Pubmed Cited)">
##INFO=<ID=TPA,Number=0,Type=Flag,Description="Provisional Third Party Annotation(TPA) (currently rs from PHARMGKB who will give phenotype data)">
##INFO=<ID=PMC,Number=0,Type=Flag,Description="Links exist to PubMed Central article">
##INFO=<ID=S3D,Number=0,Type=Flag,Description="Has 3D structure - SNP3D table">
##INFO=<ID=SLO,Number=0,Type=Flag,Description="Has SubmitterLinkOut - From SNP->SubSNP->Batch.link_out">
##INFO=<ID=NSF,Number=0,Type=Flag,Description="Has non-synonymous frameshift A coding region variation where one allele in the set changes all downstream amino acids. FxnClass = 44">
##INFO=<ID=NSM,Number=0,Type=Flag,Description="Has non-synonymous missense A coding region variation where one allele in the set changes protein peptide. FxnClass = 42">
##INFO=<ID=NSN,Number=0,Type=Flag,Description="Has non-synonymous nonsense A coding region variation where one allele in the set changes to STOP codon (TER). FxnClass = 41">
##INFO=<ID=REF,Number=0,Type=Flag,Description="Has reference A coding region variation where one allele in the set is identical to the reference sequence. FxnCode = 8">
##INFO=<ID=SYN,Number=0,Type=Flag,Description="Has synonymous A coding region variation where one allele in the set does not change the encoded amino acid. FxnCode = 3">
##INFO=<ID=U3,Number=0,Type=Flag,Description="In 3' UTR Location is in an untranslated region (UTR). FxnCode = 53">
##INFO=<ID=U5,Number=0,Type=Flag,Description="In 5' UTR Location is in an untranslated region (UTR). FxnCode = 55">
##INFO=<ID=ASS,Number=0,Type=Flag,Description="In acceptor splice site FxnCode = 73">
##INFO=<ID=DSS,Number=0,Type=Flag,Description="In donor splice-site FxnCode = 75">
##INFO=<ID=INT,Number=0,Type=Flag,Description="In Intron FxnCode = 6">
##INFO=<ID=R3,Number=0,Type=Flag,Description="In 3' gene region FxnCode = 13">
##INFO=<ID=R5,Number=0,Type=Flag,Description="In 5' gene region FxnCode = 15">
##INFO=<ID=OTH,Number=0,Type=Flag,Description="Has other variant with exactly the same set of mapped positions on NCBI refernce assembly.">
##INFO=<ID=CFL,Number=0,Type=Flag,Description="Has Assembly conflict. This is for weight 1 and 2 variant that maps to different chromosomes on different assemblies.">
##INFO=<ID=ASP,Number=0,Type=Flag,Description="Is Assembly specific. This is set if the variant only maps to one assembly">
##INFO=<ID=MUT,Number=0,Type=Flag,Description="Is mutation (journal citation, explicit fact): a low frequency variation that is cited in journal and other reputable sources">
##INFO=<ID=VLD,Number=0,Type=Flag,Description="Is Validated.  This bit is set if the variant has 2+ minor allele count based on frequency or genotype data.">
##INFO=<ID=G5A,Number=0,Type=Flag,Description=">5% minor allele frequency in each and all populations">
##INFO=<ID=G5,Number=0,Type=Flag,Description=">5% minor allele frequency in 1+ populations">
##INFO=<ID=HD,Number=0,Type=Flag,Description="Marker is on high density genotyping kit (50K density or greater).  The variant may have phenotype associations present in dbGaP.">
##INFO=<ID=GNO,Number=0,Type=Flag,Description="Genotypes available. The variant has individual genotype (in SubInd table).">
##INFO=<ID=KGValidated,Number=0,Type=Flag,Description="1000 Genome validated">
##INFO=<ID=KGPhase1,Number=0,Type=Flag,Description="1000 Genome phase 1 (incl. June Interim phase 1)">
##INFO=<ID=KGPilot123,Number=0,Type=Flag,Description="1000 Genome discovery all pilots 2010(1,2,3)">
##INFO=<ID=KGPROD,Number=0,Type=Flag,Description="Has 1000 Genome submission">
##INFO=<ID=OTHERKG,Number=0,Type=Flag,Description="non-1000 Genome submission">
##INFO=<ID=PH3,Number=0,Type=Flag,Description="HAP_MAP Phase 3 genotyped: filtered, non-redundant">
##INFO=<ID=CDA,Number=0,Type=Flag,Description="Variation is interrogated in a clinical diagnostic assay">
##INFO=<ID=LSD,Number=0,Type=Flag,Description="Submitted from a locus-specific database">
##INFO=<ID=MTP,Number=0,Type=Flag,Description="Microattribution/third-party annotation(TPA:GWAS,PAGE)">
##INFO=<ID=OM,Number=0,Type=Flag,Description="Has OMIM/OMIA">
##INFO=<ID=NOC,Number=0,Type=Flag,Description="Contig allele not present in variant allele list. The reference sequence allele at the mapped position is not present in the variant allele list, adjusted for orientation.">
##INFO=<ID=WTD,Number=0,Type=Flag,Description="Is Withdrawn by submitter If one member ss is withdrawn by submitter, then this bit is set.  If all member ss' are withdrawn, then the rs is deleted to SNPHistory">
##INFO=<ID=NOV,Number=0,Type=Flag,Description="Rs cluster has non-overlapping allele sets. True when rs set has more than 2 alleles from different submissions and these sets share no alleles in common.">
##FILTER=<ID=NC,Description="Inconsistent Genotype Submission For At Least One Sample">
##INFO=<ID=CAF,Number=.,Type=String,Description="An ordered, comma delimited list of allele frequencies based on 1000Genomes, starting with the reference allele followed by alternate alleles as ordered in the ALT column. Where a 1000Genomes alternate allele is not in the dbSNPs alternate allele set, the allele is added to the ALT column.  The minor allele is the second largest value in the list, and was previuosly reported in VCF as the GMAF.  This is the GMAF reported on the RefSNP and EntrezSNP pages and VariationReporter">
##INFO=<ID=COMMON,Number=1,Type=Integer,Description="RS is a common SNP.  A common SNP is one that has at least one 1000Genomes population with a minor allele of frequency >= 1% and for which 2 or more founders contribute to that minor allele frequency.">
##INFO=<ID=OLD_VARIANT,Number=.,Type=String,Description="Original chr:pos:ref:alt encoding">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
1\t10019\trs376643643\tTA\tT\t.\t.\tRS=376643643;RSPOS=10020;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10055\trs373328635\tT\tTA\t.\t.\tRS=373328635;RSPOS=10056;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10108\trs62651026\tC\tT\t.\t.\tRS=62651026;RSPOS=10108;dbSNPBuildID=129;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10109\trs376007522\tA\tT\t.\t.\tRS=376007522;RSPOS=10109;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10139\trs368469931\tA\tT\t.\t.\tRS=368469931;RSPOS=10139;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10144\trs144773400\tTA\tT\t.\t.\tRS=144773400;RSPOS=10145;dbSNPBuildID=134;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10146\trs375931351\tAC\tA\t.\t.\tRS=375931351;RSPOS=10147;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10150\trs371194064\tC\tT\t.\t.\tRS=371194064;RSPOS=10150;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10177\trs367896724\tA\tAC\t.\t.\tRS=367896724;RSPOS=10177;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005100002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10177\trs201752861\tA\tC\t.\t.\tRS=201752861;RSPOS=10177;dbSNPBuildID=137;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10180\trs201694901\tT\tC\t.\t.\tRS=201694901;RSPOS=10180;dbSNPBuildID=137;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10228\trs143255646\tTA\tT\t.\t.\tRS=143255646;RSPOS=10229;dbSNPBuildID=134;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10228\trs200462216\tTAACCCCTAACCCTAACCCTAAACCCTA\tT\t.\t.\tRS=200462216;RSPOS=10229;dbSNPBuildID=137;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10230\trs376846324\tAC\tA\t.\t.\tRS=376846324;RSPOS=10231;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10231\trs200279319\tC\tA\t.\t.\tRS=200279319;RSPOS=10231;dbSNPBuildID=137;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10234\trs145599635\tC\tT\t.\t.\tRS=145599635;RSPOS=10234;dbSNPBuildID=134;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10235\trs540431307\tT\tTA\t.\t.\tRS=540431307;RSPOS=10235;dbSNPBuildID=142;SSR=0;SAO=0;VP=0x050000020005000000000200;WGT=1;VC=DIV;R5;ASP
1\t10248\trs148908337\tA\tT\t.\t.\tRS=148908337;RSPOS=10248;dbSNPBuildID=134;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10249\trs375044980\tAAC\tA\t.\t.\tRS=375044980;RSPOS=10250;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10250\trs199706086\tA\tC\t.\t.\tRS=199706086;RSPOS=10250;dbSNPBuildID=137;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10254\trs140194106\tTA\tT\t.\t.\tRS=140194106;RSPOS=10255;dbSNPBuildID=134;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10257\trs111200574\tA\tC\t.\t.\tRS=111200574;RSPOS=10257;dbSNPBuildID=132;SSR=0;SAO=0;VP=0x050100020005000102000100;WGT=1;VC=SNV;SLO;R5;ASP;GNO;OTHERKG
1\t10259\trs200940095\tC\tA\t.\t.\tRS=200940095;RSPOS=10259;dbSNPBuildID=137;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10291\trs145427775\tC\tT\t.\t.\tRS=145427775;RSPOS=10291;dbSNPBuildID=134;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10327\trs112750067\tT\tC\t.\t.\tRS=112750067;RSPOS=10327;dbSNPBuildID=132;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10328\trs201106462\tAACCCCTAACCCTAACCCTAACCCT\tA\t.\t.\tRS=201106462;RSPOS=10329;dbSNPBuildID=137;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10329\trs150969722\tAC\tA\t.\t.\tRS=150969722;RSPOS=10330;dbSNPBuildID=134;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10352\trs145072688\tT\tTA\t.\t.\tRS=145072688;RSPOS=10353;dbSNPBuildID=134;SSR=0;SAO=0;VP=0x050000020015000002000200;WGT=1;VC=DIV;R5;OTH;ASP;OTHERKG
1\t10383\trs147093981\tA\tAC\t.\t.\tRS=147093981;RSPOS=10383;dbSNPBuildID=134;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10389\trs373144384\tAC\tA\t.\t.\tRS=373144384;RSPOS=10390;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10433\trs56289060\tA\tAC\t.\t.\tRS=56289060;RSPOS=10433;dbSNPBuildID=129;SSR=0;SAO=0;VP=0x050000020005000002000200;WGT=1;VC=DIV;R5;ASP;OTHERKG
1\t10439\trs112766696\tAC\tA\t.\t.\tRS=112766696;RSPOS=10440;dbSNPBuildID=132;SSR=0;SAO=0;VP=0x050100020005000102000200;WGT=1;VC=DIV;SLO;R5;ASP;GNO;OTHERKG
1\t10440\trs112155239\tC\tA\t.\t.\tRS=112155239;RSPOS=10440;dbSNPBuildID=132;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10469\trs370233998\tC\tG\t.\t.\tRS=370233998;RSPOS=10469;dbSNPBuildID=138;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10478\trs528916756\tC\tG\t.\t.\tRS=528916756;RSPOS=10478;dbSNPBuildID=142;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10483\trs547662686\tC\tT\t.\t.\tRS=547662686;RSPOS=10483;dbSNPBuildID=142;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10490\trs565971701\tG\tA\t.\t.\tRS=565971701;RSPOS=10490;dbSNPBuildID=142;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
1\t10492\trs55998931\tC\tT\t.\t.\tRS=55998931;RSPOS=10492;dbSNPBuildID=129;SSR=0;SAO=0;VP=0x050000020005000002000100;WGT=1;VC=SNV;R5;ASP;OTHERKG
3\t60069\trs549251461\tC\tT\t.\t.\tRS=549251461;RSPOS=60069;dbSNPBuildID=142;SSR=0;SAO=0;VP=0x050000000005000000000100;WGT=1;VC=SNV;ASP
3\t60079\trs567712286\tA\tG\t.\t.\tRS=567712286;RSPOS=60079;dbSNPBuildID=142;SSR=0;SAO=0;VP=0x050000000005000000000100;WGT=1;VC=SNV;ASP
3\t60157\trs186476240\tG\tA\t.\t.\tRS=186476240;RSPOS=60157;dbSNPBuildID=135;SSR=0;SAO=0;VP=0x050000000005000016000100;WGT=1;VC=SNV;ASP;KGPhase1;KGPROD;OTHERKG
3\t60189\trs550163140\tA\tG\t.\t.\tRS=550163140;RSPOS=60189;dbSNPBuildID=142;SSR=0;SAO=0;VP=0x050000000005000000000100;WGT=1;VC=SNV;ASP
3\t60194\trs544368664\tC\tT\t.\t.\tRS=544368664;RSPOS=60194;dbSNPBuildID=142;SSR=0;SAO=0;VP=0x050000000005000002000100;WGT=1;VC=SNV;ASP;OTHERKG
3\t60197\trs115479960\tG\tA\t.\t.\tRS=115479960;RSPOS=60197;dbSNPBuildID=132;SSR=0;SAO=0;VP=0x050000000005130016000100;WGT=1;VC=SNV;ASP;G5A;G5;KGPhase1;KGPROD;OTHERKG
3\t60201\trs539368937\tT\tC\t.\t.\tRS=539368937;RSPOS=60201;dbSNPBuildID=142;SSR=0;SAO=0;VP=0x050000000005000000000100;WGT=1;VC=SNV;ASP
3\t60202\trs28729284\tC\tG\t.\t.\tRS=28729284;RSPOS=60202;dbSNPBuildID=125;SSR=0;SAO=0;VP=0x050100000005000116000100;WGT=1;VC=SNV;SLO;ASP;GNO;KGPhase1;KGPROD;OTHERKG
3\t60316\trs190125143\tC\tA,G\t.\t.\tRS=190125143;RSPOS=60316;dbSNPBuildID=135;SSR=0;SAO=0;VP=0x050000000005000016000100;WGT=1;VC=SNV;ASP;KGPhase1;KGPROD;OTHERKG
3\t60322\trs566180130\tG\tA\t.\t.\tRS=566180130;RSPOS=60322;dbSNPBuildID=142;SSR=0;SAO=0;VP=0x050000000005000000000100;WGT=1;VC=SNV;ASP
"""
test_var_chr1 = [
    [1, 10100, "A", "G"],  # Not in dbSNP
    [1, 10139, "A", "T"],  # rs368469931
    [1, 10250, "A", "C"],  # rs199706086
    [1, 10259, "C", "A"],  # rs200940095
]
test_var_chr3 = [
    [3, 60069, "A", "T"],  # ref mismatch
    [3, 60079, "AG", "A"],  # ref mismatch
    [3, 60322, "G", "A"],  # rs566180130
]


def test_match_ids(tmp_path):
    dbsnp_path = Path(tmp_path) / "dbsnp.vcf"
    compressed_writer = CompressedTextWriter(dbsnp_path)

    with compressed_writer as file_handle:
        file_handle.write(dbsnp_file_contents)

    columns = ["CHROM", "POS", "REF", "ALT"]
    df1 = pd.DataFrame(test_var_chr1, columns=columns)
    df3 = pd.DataFrame(test_var_chr3, columns=columns)

    dfs = [df1, df3]
    result = match_ids(dfs, dbsnp_path)

    assert len(result) == len(test_var_chr3) + len(
        test_var_chr1
    ), "Number of results doesn't match number of input variants"
    assert result == [
        "1:10100:A:G",
        "rs368469931",
        "rs199706086",
        "rs200940095",
        "3:60069:A:T",
        "3:60079:AG:A",
        "rs566180130",
    ]
