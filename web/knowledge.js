const $=id=>document.getElementById(id);
let records=[], limit=20;
function render() {
  const query=$("record-search").value.toLowerCase().trim(), category=$("record-category").value;
  const matches=records.filter(r=>(!category||r.category===category)&&(!query||(r.title+" "+r.content).toLowerCase().includes(query)));
  $("record-count").textContent=`${matches.length.toLocaleString()} matching records · showing ${Math.min(limit,matches.length)}`;
  $("record-list").replaceChildren();
  for(const record of matches.slice(0,limit)) {
    const article=document.createElement("article");article.className="knowledge-record";
    const heading=document.createElement("h2");heading.textContent=record.title;
    const meta=document.createElement("p");meta.className="caption";meta.textContent=`${record.category} / ${record.subcategory} · ${record.record_type.replaceAll('_',' ')}`;
    const content=document.createElement("pre");content.className="record-content";content.textContent=record.content;
    const link=document.createElement("a");link.textContent="View source";link.href=record.source_url;link.target="_blank";link.rel="noopener";
    article.append(heading,meta,content,link);$("record-list").append(article);
  }
  $("more-records").hidden=limit>=matches.length;
}
for(const id of ["record-search","record-category"]) $(id).addEventListener("input",()=>{limit=20;render();});
$("more-records").onclick=()=>{limit+=20;render();};
try {
  const response=await fetch("index.json");if(!response.ok)throw new Error("HTTP "+response.status);
  const index=await response.json();records=index.records;
  const m=index.manifest;
  $("corpus-summary").textContent=`${m.records.toLocaleString()} records: ${m.record_types.original_reference} original references, ${m.record_types.lesson_passage+m.record_types.code} sourced lesson/code passages, ${m.record_types.generated_worked_example} generated worked calculations.`;
  for(const [category,count] of Object.entries(m.categories)) {const option=document.createElement("option");option.value=category;option.textContent=`${category} (${count})`;$("record-category").append(option);}
  render();
} catch(error) {$("corpus-summary").textContent="Could not load records: "+error.message;}
