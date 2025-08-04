"""
Enhanced Chunking Demo
Demonstrates the improved document chunking capabilities
"""

import time
import json
from typing import Dict, Any

def demo_enhanced_chunking():
    """Demonstrate enhanced chunking capabilities"""
    
    print("🚀 Enhanced Document Chunking Demo")
    print("=" * 50)
    
    # Sample policy text for demonstration
    sample_text = """
    INSURANCE POLICY DOCUMENT
    
    SECTION 1: DEFINITIONS
    In this policy, the following terms have the meanings set forth below:
    "Insured" means the person or entity named in the declarations page.
    "Policy Period" means the period of time for which this policy provides coverage.
    "Premium" means the amount charged for insurance coverage under this policy.
    
    SECTION 2: COVERAGE PROVISIONS
    This policy provides coverage for direct physical loss or damage to covered property
    when such loss or damage is caused by a covered peril, subject to all terms,
    conditions, and exclusions of this policy.
    
    Coverage A - Dwelling: We cover the dwelling on the residence premises shown in the
    declarations, including structures attached to the dwelling.
    
    Coverage B - Other Structures: We cover other structures on the residence premises
    set apart from the dwelling by clear space.
    
    SECTION 3: EXCLUSIONS
    We do not cover loss or damage caused directly or indirectly by:
    1. Ordinance or Law
    2. Earth Movement
    3. Water damage from flooding
    4. Power failure that occurs off the residence premises
    5. Neglect of the insured to use all reasonable means to save and preserve property
    
    SECTION 4: CONDITIONS
    The following conditions apply to this policy:
    
    Duties After Loss: In case of a loss to covered property, you must:
    - Give immediate notice to us or our agent
    - Protect the property from further damage
    - Prepare an inventory of damaged personal property
    - Submit a proof of loss within 60 days after our request
    
    SECTION 5: CLAIM PROCEDURES
    To file a claim under this policy, the insured must:
    1. Report the claim immediately to the insurance company
    2. Provide documentation of the loss
    3. Cooperate with the claims investigation
    4. Submit required forms and documentation
    
    The company reserves the right to inspect the damaged property and may require
    an examination under oath of the insured regarding the claim.
    """
    
    print(f"📄 Sample document: {len(sample_text)} characters")
    print()
    
    # Test different chunking strategies
    strategies_to_test = [
        ("Original Semantic", "semantic_split"),
        ("Enhanced Adaptive", "adaptive_split"), 
        ("Lightning Fast", "lightning_fast_split"),
        ("Policy Aware", "smart_fast_split"),
        ("Enhanced Framework", "enhanced_framework")
    ]
    
    results = {}
    
    for strategy_name, strategy_type in strategies_to_test:
        print(f"🧪 Testing {strategy_name}...")
        
        try:
            start_time = time.time()
            
            if strategy_type == "enhanced_framework":
                # Test the complete enhanced framework
                try:
                    from utils.enhanced_chunking import (
                        EnhancedChunkingFramework, 
                        ChunkingConfig, 
                        ChunkingStrategy
                    )
                    
                    config = ChunkingConfig(
                        chunk_size=400,  # Smaller for demo
                        chunk_overlap=50,
                        strategy=ChunkingStrategy.ADAPTIVE
                    )
                    
                    framework = EnhancedChunkingFramework(config)
                    result = framework.chunk_document(sample_text)
                    
                    chunks = result['chunks']
                    quality_metrics = result.get('quality_metrics', {})
                    
                    execution_time = time.time() - start_time
                    
                    print(f"   ✅ {len(chunks)} chunks created in {execution_time:.3f}s")
                    print(f"   📊 Quality score: {quality_metrics.get('content_preservation_score', 0):.3f}")
                    print(f"   📏 Avg length: {quality_metrics.get('avg_chunk_length', 0):.1f}")
                    
                    results[strategy_name] = {
                        'chunks': len(chunks),
                        'time': execution_time,
                        'quality': quality_metrics,
                        'sample_chunk': chunks[0][:100] + "..." if chunks else "No chunks"
                    }
                    
                except ImportError:
                    print("   ⚠️ Enhanced framework not available")
                    continue
                    
            else:
                # Test basic strategies
                from utils.splitter import semantic_split
                
                if strategy_type == "semantic_split":
                    chunks = semantic_split(sample_text, chunk_size=400, chunk_overlap=50)
                elif strategy_type == "adaptive_split":
                    try:
                        from utils.splitter import adaptive_split
                        chunks = adaptive_split(sample_text, chunk_size=400, chunk_overlap=50)
                    except ImportError:
                        chunks = semantic_split(sample_text, chunk_size=400, chunk_overlap=50)
                elif strategy_type == "lightning_fast_split":
                    try:
                        from utils.splitter import lightning_fast_split
                        chunks = lightning_fast_split(sample_text, chunk_size=400, chunk_overlap=50)
                    except ImportError:
                        chunks = semantic_split(sample_text, chunk_size=400, chunk_overlap=50)
                elif strategy_type == "smart_fast_split":
                    try:
                        from utils.splitter import smart_fast_split
                        chunks = smart_fast_split(sample_text, chunk_size=400, chunk_overlap=50)
                    except ImportError:
                        chunks = semantic_split(sample_text, chunk_size=400, chunk_overlap=50)
                else:
                    chunks = semantic_split(sample_text, chunk_size=400, chunk_overlap=50)
                
                execution_time = time.time() - start_time
                
                print(f"   ✅ {len(chunks)} chunks created in {execution_time:.3f}s")
                
                results[strategy_name] = {
                    'chunks': len(chunks),
                    'time': execution_time,
                    'sample_chunk': chunks[0][:100] + "..." if chunks else "No chunks"
                }
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ Failed: {e}")
            results[strategy_name] = {
                'chunks': 0,
                'time': execution_time,
                'error': str(e)
            }
        
        print()
    
    # Display comparison
    print("📊 PERFORMANCE COMPARISON")
    print("-" * 40)
    
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if successful_results:
        # Sort by speed
        sorted_by_speed = sorted(successful_results.items(), key=lambda x: x[1]['time'])
        
        print("🏃 Speed Ranking:")
        for i, (name, result) in enumerate(sorted_by_speed):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            print(f"  {rank} {name}: {result['time']:.3f}s ({result['chunks']} chunks)")
        
        print()
        print("📝 Sample Chunks:")
        for name, result in successful_results.items():
            print(f"\n🔍 {name}:")
            print(f"   {result.get('sample_chunk', 'No sample available')}")
    
    # Save results
    with open("demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to demo_results.json")
    
    return results

if __name__ == "__main__":
    try:
        results = demo_enhanced_chunking()
        print("\n🎉 Demo completed successfully!")
        
        # Quick summary
        successful_count = sum(1 for r in results.values() if 'error' not in r)
        total_count = len(results)
        
        print(f"📈 Summary: {successful_count}/{total_count} strategies worked")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
