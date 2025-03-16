let StartingDm = 
{
  rootElement: {
    subElements : [
        {
            identifier: "A",
            order: 1,
            dependencies: []
        },
        {
            identifier: "B",
            order: 2,
            dependencies: [["C",1],["E",1]]
        },
        {
            identifier: "C",
            order: 3,
            dependencies: [["E",2]]
        },
        {
            identifier: "D",
            order: 4,
            dependencies: [["B",3],["E",2]]
        },
        {
            identifier: "E",
            order: 5,
            dependencies: [["G",3]]
        },
        {
            identifier: "F",
            order: 6,
            dependencies: [["A",2],["D",1]]
        },
        {
            identifier: "G",
            order: 7,
            dependencies: [["C",3],["F",1]]
        }
    ]
  } 
};

function generateDSM(dsmModel, parentElement) {
    // DSM BASE
    const root = dsmModel.rootElement;
    let structure = document.createElement("div");
    structure.setAttribute("class", "dsm");
    let rootTable = document.createElement("table");
    let rootTableRow = document.createElement("tr");
    let rootTableBlock = document.createElement("th");
    rootTableBlock.setAttribute("class", "rootcomponent");
    rootTableBlock.textContent = "RT";
    let layoutRow = document.createElement("div");
    layoutRow.setAttribute("class", "dsmrow");
    let layoutColSubElement = document.createElement("div");
    // layoutColSubElement.setAttribute("class", "column");
    let layoutColDependency = document.createElement("div");
    // layoutColDependency.setAttribute("class", "column");
    let subElementTable = document.createElement("table");
    let dependencyTable = document.createElement("table");

    // DSM BASE BUILD
    structure.appendChild(rootTable);
    structure.appendChild(layoutRow);
    rootTable.appendChild(rootTableRow);
    rootTableRow.appendChild(rootTableBlock);
    layoutRow.appendChild(layoutColSubElement);
    layoutRow.appendChild(layoutColDependency);
    layoutColSubElement.appendChild(subElementTable);
    layoutColDependency.appendChild(dependencyTable);

    // DSM SUBELEMENT 
    let subElements = dsmModel.rootElement.subElements;
    let countSubElements = subElements.length;

    //      SORT SUBELEMENTS BASED ON ORDER
    subElements = subElements.sort(function(a,b) {
        return a.order - b.order;
    });

    //      CELL SORTING FUNCTION
    function orderSubElementCells(tableRowElement) {
        let cells = Array.from(tableRowElement.children);
        let sortedCells = cells.sort(function(a,b) {
            return a.getAttribute("sequence") - b.getAttribute("sequence");
        });
        for (let i = 0; i < tableRowElement.children.length; i++) {
            tableRowElement.children.item(i).replaceWith(sortedCells[i]);
        }
    }

    // ADD NEW BLOCKS AND DEPENDENCY CELLS FOR EACH SUBELEMENT
    subElements.forEach(subElement => {
        // ADD A HEADER BLOCK AND SUBELEMENT BLOCK
        let newSubElementHeader = document.createElement("th");
        newSubElementHeader.setAttribute("class", "subcomponent");
        newSubElementHeader.textContent = subElement.identifier;
        rootTableRow.appendChild(newSubElementHeader);
        let newSubElementTableRow = document.createElement("tr");
        let newSubElementTableBlock = document.createElement("th");
        newSubElementTableBlock.setAttribute("class", "subcomponent");
        newSubElementTableBlock.textContent = subElement.identifier;
        newSubElementTableRow.appendChild(newSubElementTableBlock);
        subElementTable.appendChild(newSubElementTableRow);

        // ADD N DEPENDENCY CELLS
        let newDependencyTableRow = document.createElement("tr");
        for (let i = 0; i < countSubElements; i++) {
            let newCell = document.createElement("td");
            let subElementIntersect = subElements[i].identifier;
            let subElementSequence = subElements[i].order;
            newCell.setAttribute("subElementIntersect", subElementIntersect);
            newCell.setAttribute("sequence", subElementSequence);
            subElement.dependencies.forEach(dependency => {
                if (dependency[0]==subElementIntersect) {
                    newCell.textContent=dependency[1];
                }
            });
            if (subElement.identifier==subElementIntersect) {
                newCell.classList.add("selfIntersectCell");
            }
            newDependencyTableRow.appendChild(newCell);
        }
        orderSubElementCells(newDependencyTableRow);
        dependencyTable.appendChild(newDependencyTableRow);
    });
    
    // APPEND TO DOCUMENT
    parentElement.appendChild(structure);
    console.log('DSM Successfully Generated.')
}